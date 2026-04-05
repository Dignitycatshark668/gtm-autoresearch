# gtm-autoresearch

**Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) loop, adapted for SaaS scoring models.**

[![Based on](https://img.shields.io/badge/Based_on-Karpathy's_Autoresearch-orange)](https://github.com/karpathy/autoresearch)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://python.org)

> Karpathy's autoresearch runs an AI agent that edits training code, measures validation loss, keeps improvements, and repeats - hundreds of experiments overnight on a single GPU. We took the same loop and pointed it at GTM scoring models: health scores, churn risk, ICP fit. Instead of `val_bpb`, we optimize for revenue retention by tier. Instead of 5-minute GPU runs, we score 5,000 accounts in 56ms. The constraints are the same: one mutable artifact, one metric, keep-or-revert, log everything.

---

## Why this exists

SaaS teams change their scoring models constantly — but almost nobody logs these changes as experiments. In ML that's malpractice. In GTM scoring it's standard practice.

A typical workflow:
1. VP of CS says "we should weight NPS higher"
2. Someone updates a config in Gainsight or a spreadsheet
3. Nobody records what the old weights were
4. Nobody measures if the change improved prediction
5. Three months later, another tweak on top
6. Nobody can tell which version was best

This is the equivalent of editing `train.py`, never committing, and deleting your terminal history.

Most teams already experiment with their scoring models — they build geo-specific variants, test different feature sets, validate against revenue. The problem isn't a lack of experimentation. It's a lack of **memory**. Every weight change is a lost experiment because nobody logs what was tried, what it replaced, or whether it actually worked.

**gtm-autoresearch** adds the memory layer. Every weight change becomes a logged, comparable, reversible experiment. The engine finds better weights and proves they work — and the journal means the next experiment learns from all the previous ones instead of starting blind.

## Where this fits

```
┌─────────────────────────────────────────────┐
│  Channel Execution Layer                    │
│  Campaign variants, HOTL governance,        │
│  multi-agent orchestration, API execution   │
│  → Described in the playbook (see Related)  │
├─────────────────────────────────────────────┤
│  Experiment Engine (THIS REPO)              │
│  Weight optimization, experiment journal,   │
│  13 GTM-native metrics, keep-or-revert     │
├─────────────────────────────────────────────┤
│  Your ICP Foundation                        │
│  Scoring model, enrichment pipeline,        │
│  segmentation, ground truth data            │
│  → You build this (or already have it)      │
└─────────────────────────────────────────────┘
```

The bottom layer is the hardest to build and the most defensible - your ICP model, your enrichment pipeline, your ground truth. This repo sits in the middle: it takes your foundation and adds experiment discipline. The channel execution layer (running campaign variants, A/B tests on email/ads/pages) sits above and is described in the [playbook](https://mazorda.com/playbooks/autonomous-gtm-experimentation), not this repo.

## What this repo does NOT do

- **Send emails or run ad campaigns.** This optimizes scoring models, not campaign assets.
- **Replace your ICP model.** You need a scoring foundation with features and known outcomes first. This makes it better.
- **Require an LLM.** Mutations are random weight perturbations - cheap, fast, and sufficient for the weight-space search.
- **Touch your production systems.** The engine reads data, runs experiments in memory, and writes to a journal file. Nothing else.

---

## How it maps to Karpathy's design

The original [autoresearch](https://github.com/karpathy/autoresearch) is a 630-line Python harness built around three constraints: one mutable file (`train.py`), one evaluation metric (`val_bpb`), and a keep-or-revert rule enforced by git. The agent runs hundreds of experiments overnight, each bounded to 5 minutes on a single GPU.

We preserved the constraints. We changed what they operate on.

| Karpathy's autoresearch | gtm-autoresearch |
|---|---|
| **Mutable artifact:** `train.py` (GPT training code) | **Mutable artifact:** scoring spec (feature weights) |
| **Experiment:** 5-min GPU training run | **Experiment:** score all accounts (56ms for 5K) |
| **Metric:** `val_bpb` (bits per byte) | **Metric:** AUC-ROC, Revenue Capture, NRR by tier |
| **Mutation:** AI edits Python code | **Mutation:** random weight perturbation |
| **Rollback:** `git reset` | **Rollback:** keep-or-revert (journal preserves all) |
| **Budget:** 1 GPU, 5 min/experiment | **Budget:** 1 CPU, <100ms/experiment |
| **Scale:** ~100 experiments/day | **Scale:** ~500 experiments/second |
| **Evaluation:** single scalar (val_bpb) | **Evaluation:** 13 GTM-native metrics (NRR, GRR, HHI...) |

### What we kept from Karpathy

1. **The evaluation harness is read-only.** The loop can change weights. It cannot change how results are measured. This is the core governance constraint that prevents metric gaming.
2. **Every experiment is logged.** Append-only journal. Full weight snapshots. Reproducible.
3. **Keep-or-revert discipline.** Only improvements survive.
4. **Speed enables exploration.** Karpathy's insight: the loop must be fast enough to run hundreds of experiments, not dozens. We vectorized scoring in numpy to hit 300-500/second.
5. **Features are frozen during runs.** Adding features is a human decision outside the loop. The agent optimizes weights on a fixed set.

### What we changed for GTM

1. **Multi-metric evaluation.** Karpathy uses one scalar. GTM scoring needs NRR, churn prediction, expansion rate, and fairness across segments - simultaneously. We optimize one primary metric but report all 13 so humans can catch proxy drift.
2. **GTM-native metrics.** Net Revenue Retention by tier, winback rate by former tier, revenue concentration (HHI). These are what RevOps teams use to make resource decisions. MLflow doesn't know what NRR is.
3. **No LLM required.** Karpathy's agents use LLMs to edit code. Our mutations are random weight perturbations - cheap, fast, and sufficient for the weight-space search. The search space (15-25 features, integer weights 0-25) is small enough that random hill-climbing explores it thoroughly in minutes.

## Validated results

### Test A: Synthetic signal recovery

Generated 5,000 accounts with **known planted signals** (some features predict retention, others are noise). Ran 2,000 experiments in 6.8 seconds. The engine recovered all planted signals in the correct rank order:

| Signal | Planted strength | Engine found | Recovered? |
|---|---|---|---|
| alert_created | Strongest | Weight 25 (highest) | Yes |
| session_depth | 2nd | Weight 23 | Yes |
| activation_speed | 3rd | Weight 20 | Yes |
| has_exported | Moderate | Weight 10 | Yes |
| credit_limit_hits | Weak positive | Weight 6 (positive) | Yes |
| noise_feature_1 | Zero | Weight 1 (near zero) | Yes |
| noise_feature_2 | Zero | Weight 1 (near zero) | Yes |

**AUC-ROC:** 0.63 -> 0.75 | **Tier separation:** 1.38 -> 1.85 | **Time:** 6.8 seconds

```
# Reproduce this yourself:
python3 examples/health_score/generate_data.py
python3 engine/scoring_engine.py --n 2000 --metric auc_roc_retain
```

### Test B: Production validation

Also tested against a production B2B SaaS dataset (tens of thousands of accounts, multiple data sources, known retention and revenue outcomes). The engine found signal hierarchies that manual weight-tuning missed - including cases where signals assumed to be negative (user friction events) turned out to be strong positive engagement indicators. Details available on request.

## Quick start

### Option A: Run on your own data (recommended)

If you already have account data with known outcomes (retention, churn, revenue):

```bash
# 1. Prepare a parquet file with columns: your scoring features + outcomes
#    Required outcome columns (at least one): retained_6mo, churned, actual_ltv, mrr
#    Feature columns: everything else (the engine auto-detects them)

# 2. Run 2,000 experiments (~7 seconds)
python3 engine/scoring_engine.py --n 2000 --metric revenue_capture_at_20 --data your_ground_truth.parquet

# 3. Review what the engine found
python3 analysis/review.py experiment_journal.jsonl
```

### Option B: Try with synthetic data first

No data yet? Generate a test dataset with planted signals to see how the engine works:

```bash
# 1. Generate synthetic data (5K accounts, planted signals)
python3 examples/health_score/generate_data.py

# 2. Run 2,000 experiments (~7 seconds)
python3 engine/scoring_engine.py --n 2000 --metric auc_roc_retain

# 3. Review results - the engine should recover all planted signals
python3 analysis/review.py experiment_journal.jsonl

# 4. Run validation tests
python3 tests/test_engine.py
```

## How it works

### 1. Define a scoring spec

```yaml
model: health_score
max_score: 100

features:
  alert_created:
    weight: 5       # Starting weight (the engine will optimize this)
    min: 0
    max: 25
  session_depth:
    weight: 5
    min: 0
    max: 25
  # ... your features

tiers:
  A: {min: 60, max: 101}
  B: {min: 35, max: 60}
  C: {min: 15, max: 35}
  D: {min: 0, max: 15}
```

Or skip the spec - the engine auto-detects features from your parquet columns.

### 2. Prepare ground truth

A parquet file with one row per account. Columns: features + outcomes (`retained_6mo`, `churned`, `actual_ltv`, `mrr`).

### 3. Run experiments

```bash
python3 engine/scoring_engine.py \
    --n 2000 \
    --metric auc_roc_retain \
    --data your_ground_truth.parquet
```

Available metrics: `auc_roc_retain`, `auc_roc_churn`, `revenue_capture_at_20`, `tier_separation`

### 4. Review the journal

Every experiment produces a JSONL entry:

```json
{
  "experiment_id": "exp_00034",
  "metric": "auc_roc_retain",
  "baseline_value": 0.6492,
  "variant_value": 0.6610,
  "outcome": "win",
  "changes": ["alert_created: 5 -> 12", "session_depth: 5 -> 8"],
  "weights_snapshot": {"alert_created": 12, "session_depth": 8},
  "metrics": {
    "auc_roc_retain": 0.6610,
    "nrr_by_tier": {"A": 1.0, "B": 0.92, "C": 0.59, "D": 0.32},
    "revenue_concentration_hhi": 3953
  }
}
```

## GTM-native evaluation metrics

This is what separates it from MLflow or generic experiment tracking. Every metric here is something a RevOps or CS team uses to make resource allocation decisions.

| Metric | What it answers |
|---|---|
| **NRR by tier** | Do higher-scored accounts retain more revenue? |
| **GRR by tier** | Pure retention without expansion masking it |
| **Logo retention by tier** | Account count retention (not revenue-weighted) |
| **Expansion rate by tier** | Do higher-scored accounts upsell? |
| **Contraction rate by tier** | Who's downgrading? (catches the "retain logo, lose revenue" pattern) |
| **Quick ratio by tier** | Growth efficiency per segment |
| **Winback rate by former tier** | Do historically high-scored churned accounts return? |
| **Activation rate by tier** | Do higher tiers activate faster? |
| **ARPA by tier** | Are tiers just proxies for plan price? |
| **Revenue concentration (HHI)** | Is too much MRR in one tier? (guardrail against gaming) |
| **Revenue Capture @20%** | Does the top 20% of scored accounts contain the most revenue? |
| **AUC-ROC** | Binary classification accuracy for retention/churn |
| **Tier separation** | LTV ratio between top and bottom quartiles |

See [docs/metrics.md](docs/metrics.md) for detailed definitions and benchmarks.

## Architecture

```
gtm-autoresearch/
├── README.md
├── LICENSE
├── engine/
│   └── scoring_engine.py    # Core: score + evaluate + experiment loop
├── analysis/
│   └── review.py            # Journal analysis + comparison
├── examples/
│   ├── health_score/
│   │   └── generate_data.py # Synthetic data: retention signals
│   └── churn_risk/
│       └── generate_data.py # Synthetic data: churn decay signals
├── spec/
│   └── example_spec.json    # Example scoring spec (output of a run)
├── docs/
│   ├── methodology.md       # Full Karpathy adaptation explained
│   └── metrics.md           # All 13 evaluation metrics defined
└── tests/
    └── test_engine.py       # 4 validation tests including signal recovery
```

## Requirements

- Python 3.9+
- numpy, pandas, pyarrow
- scikit-learn (for AUC-ROC)

## License

MIT

## Related

- [Autonomous GTM Experimentation](https://mazorda.com/playbooks/autonomous-gtm-experimentation) - the Mazorda playbook that explains when and how to apply this pattern
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) - the original ML experimentation loop this project is based on

## Credits

- **[Andrej Karpathy](https://github.com/karpathy)** - for [autoresearch](https://github.com/karpathy/autoresearch), the autonomous experiment loop that this project adapts for GTM
- **[Anthropic](https://anthropic.com)** - for [Claude Code](https://docs.anthropic.com/en/docs/claude-code), used throughout development

## Author

**[Yaniv Mazor](https://mazorda.com)** - Founder, Mazorda. GTM engineering for B2B SaaS.

Built from production work with real SaaS account data. The pattern is reusable for any SaaS company with behavioral data and known outcomes.
