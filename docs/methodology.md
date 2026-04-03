# Methodology

## The autoresearch pattern

Andrej Karpathy released [autoresearch](https://github.com/karpathy/autoresearch) in early 2026 - a minimal harness for autonomous AI experimentation. The core loop:

1. Load a fixed evaluation dataset
2. Edit a single mutable artifact (training code)
3. Run a bounded experiment
4. Measure against a single metric
5. Keep the change if improved, revert if not
6. Repeat

The design is deliberately constrained: one mutable file, one GPU, one metric, fixed time budget per experiment. This makes experiments comparable and keeps diffs small enough for human review.

## How we adapted it for GTM scoring

In Karpathy's version, the mutable artifact is `train.py` (a neural network training script). In ours, it's a **scoring spec** - a declarative definition of feature weights, tier boundaries, and evaluation criteria.

| Karpathy's loop | GTM scoring loop |
|---|---|
| Mutable artifact: `train.py` | Mutable artifact: `scoring_spec.yaml` |
| Experiment: 5-min GPU training run | Experiment: score all accounts (50-100ms) |
| Metric: val_bpb (bits per byte) | Metric: AUC-ROC, Revenue Capture, NRR by tier |
| Mutation: AI edits Python code | Mutation: random weight perturbation |
| Budget: 1 GPU, 5 min per experiment | Budget: 1 CPU, <100ms per experiment |
| Scale: ~100 experiments/day | Scale: ~500 experiments/second |

### What we kept

1. **The evaluation harness is read-only.** The loop cannot change how results are measured. Only humans can modify the evaluation function. This prevents metric gaming.
2. **Every experiment is logged.** Append-only journal with full weight snapshots. Reproducible.
3. **Keep-or-revert discipline.** Only improvements survive. No cargo-cult changes.
4. **Speed enables exploration.** 500 experiments/second means the engine explores the weight space thoroughly in seconds, not weeks.

### What we changed

1. **Multi-metric evaluation.** Karpathy uses one scalar metric. GTM scoring needs NRR, churn prediction, revenue concentration, tier separation, and fairness metrics - simultaneously. We optimize one primary metric but report all metrics so humans can catch proxy drift.
2. **GTM-native metrics.** Net Revenue Retention by tier, logo retention, expansion rate, contraction rate, winback rate, revenue concentration (HHI). These are what RevOps teams actually use to make decisions.
3. **No LLM in the loop (by default).** Karpathy's agents use LLMs to edit code. Our mutations are random weight perturbations - cheap, fast, and sufficient for the weight-space search. LLM-guided hypothesis generation is optional.

## Why weighted linear scoring, not ML

Health scores, ICP models, and engagement scores at most SaaS companies are weighted linear sums. Not neural nets, not gradient-boosted trees. Weighted sums.

This isn't because linear models are optimal. It's because:
- **Interpretability matters.** A sales rep needs to understand why an account scored 72. "Alert creation is 22 points, session depth is 15" is actionable. "The model said so" isn't.
- **Governance requires transparency.** When CS leadership asks "why did we prioritize account X over account Y," you need a human-readable answer.
- **The search space is small.** 15-25 features with integer weights 0-25 is a search space that random mutation explores thoroughly in minutes.

The autoresearch loop finds better weights for this linear model. If you need non-linear interactions, use the optimized weights as features in a logistic regression or gradient boost. The loop gives you the feature importance ranking - the ML model handles the interactions.

## Evaluation metrics explained

See [metrics.md](metrics.md) for detailed definitions of all GTM-native metrics.
