# Evaluation Metrics

Every metric in the evaluator is something a RevOps or CS team uses to make resource allocation decisions. MLflow tracks loss curves. We track revenue retention by tier.

## Primary metrics (optimize for one)

### Revenue Capture @20%
What percentage of total MRR sits in the top 20% of scored accounts? Random scoring = 20%. A good model concentrates revenue in the top tier.

### AUC-ROC (Retention / Churn)
Binary classification accuracy. Can the score separate accounts that retained from those that churned? 0.5 = random, 0.7+ = useful, 0.8+ = strong.

## GTM-native metrics (reported, not optimized)

### NRR by tier (Net Revenue Retention)
For each scoring tier: what percentage of starting MRR is retained after N months? Includes expansion. The metric that drives SaaS valuations.

Benchmark: healthy B2B SaaS targets 100-120%+ annual NRR.

### GRR by tier (Gross Revenue Retention)
Same as NRR but excludes expansion. Pure retention signal - no upsell masking.

Benchmark: 85-95% annual for healthy B2B SaaS.

### Logo retention by tier
Percentage of accounts (not revenue) retained. Catches cases where many small accounts churn but large accounts mask it in NRR.

### Expansion rate by tier
Percentage of retained accounts that increased their MRR. Validates that the score predicts growth potential, not just retention.

### Contraction rate by tier
Percentage of retained accounts that decreased their MRR. Catches the "A-tier paradox" - high logo retention but dropping revenue due to plan downgrades.

### Quick ratio by tier
Expansion / (Churn + Contraction) per tier. Growth efficiency metric. Tells you where to invest CS and sales resources.

### Winback rate by former tier
Of churned accounts, what percentage came back - segmented by their tier BEFORE churning? Validates the score's predictive value beyond the active lifecycle.

### Activation rate by tier
Percentage of accounts that hit a product milestone, by tier. Links scoring to time-to-value.

### ARPA by tier
Average Revenue Per Account per tier. Guardrail: if ARPA is the same across tiers, the model might just be sorting by MRR instead of behavior.

## Guardrails (constraints, not objectives)

### Tier separation
LTV ratio between top-quartile and bottom-quartile scored accounts. Should be > 2.0. If it's close to 1.0, the score isn't differentiating.

### Revenue concentration (Herfindahl Index)
HHI = sum of (tier MRR share)^2. Range: 2000 (evenly distributed across 5 tiers) to 10000 (all revenue in one tier).

- HHI < 2500: well distributed
- HHI 2500-4000: moderate concentration
- HHI > 4000: one tier dominates - the model may just be learning "high MRR = high score"

This is the key guardrail against proxy metric gaming. A model that "improves" Revenue Capture by concentrating everything in one tier will trigger a high HHI warning.
