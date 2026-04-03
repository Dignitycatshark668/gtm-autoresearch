"""Generate synthetic SaaS account data for churn risk scoring.

Different from health_score example: optimizes for churn prediction (AUC-ROC),
not retention. Features emphasize decay signals and payment health.

Planted signals:
  1. recency_drop (sessions declining) -> strong churn predictor
  2. payment_failures -> strong churn predictor
  3. days_since_login (high = bad) -> moderate churn predictor
  4. onboarding_complete (low = churn risk) -> moderate predictor
  5. noise_feature -> no predictive power

Run: python3 generate_data.py
Output: synthetic_churn_ground_truth.parquet
"""

import numpy as np
import pandas as pd

np.random.seed(99)

N = 5000

accounts = pd.DataFrame({
    "account_id": [f"churn_{i:05d}" for i in range(N)],

    # Behavioral decay signals
    "recency_drop": np.clip(np.random.beta(2, 5, N), 0, 1),        # Low = going cold
    "session_trend": np.clip(np.random.normal(0.5, 0.2, N), 0, 1), # < 0.5 = declining
    "days_since_login": np.clip(np.random.exponential(30, N), 0, 365) / 365,  # Normalized 0-1

    # Payment health
    "payment_failures": np.clip(np.random.exponential(0.1, N), 0, 1),
    "months_since_payment": np.clip(np.random.exponential(0.05, N), 0, 1),

    # Onboarding and engagement
    "onboarding_complete": np.clip(np.random.beta(5, 2, N), 0, 1),  # Most complete it
    "feature_breadth": np.clip(np.random.beta(3, 4, N), 0, 1),
    "support_tickets": np.clip(np.random.exponential(0.15, N), 0, 1),

    # Context
    "billing_monthly": np.random.binomial(1, 0.6, N).astype(float),  # Monthly churns faster
    "mrr": np.clip(np.random.lognormal(5.0, 0.8, N), 50, 5000),

    # Noise
    "noise_feature": np.random.uniform(0, 1, N),
})

# Churn probability driven by decay + payment signals
churn_logit = (
    2.5 * (1 - accounts["recency_drop"])       # Low recency = high churn
    + 2.0 * accounts["payment_failures"]         # Payment problems = churn
    + 1.5 * accounts["days_since_login"]         # Haven't logged in = churn
    + 1.0 * (1 - accounts["onboarding_complete"]) # Didn't onboard = churn
    + 0.8 * accounts["billing_monthly"]           # Monthly plans churn more
    + 0.5 * accounts["support_tickets"]           # High tickets = frustrated
    - 0.3 * accounts["feature_breadth"]           # Using features = stays
    + 0.0 * accounts["noise_feature"]             # No effect
    + 0.0 * accounts["mrr"]                       # No effect on churn
    - 2.0                                          # Intercept
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
accounts["churned"] = np.random.binomial(1, churn_prob, N).astype(float)
accounts["retained_6mo"] = 1.0 - accounts["churned"]

# LTV
base_tenure = np.where(accounts["retained_6mo"] == 1,
                       np.clip(np.random.lognormal(2.5, 0.5, N), 3, 60),
                       np.clip(np.random.lognormal(1.2, 0.8, N), 1, 12))
accounts["tenure_months"] = base_tenure
accounts["actual_ltv"] = accounts["mrr"] * accounts["tenure_months"]

# Summary
print(f"Generated {N} synthetic churn-risk accounts")
print(f"\nChurn rate: {accounts['churned'].mean():.1%}")
print(f"Retention rate: {accounts['retained_6mo'].mean():.1%}")
print(f"Mean LTV: ${accounts['actual_ltv'].mean():,.0f}")

print(f"\n--- Signal verification ---")
for feat in ["recency_drop", "payment_failures", "days_since_login",
             "onboarding_complete", "billing_monthly", "noise_feature"]:
    if accounts[feat].nunique() <= 2:
        churn_with = accounts.loc[accounts[feat] > 0, "churned"].mean()
        churn_without = accounts.loc[accounts[feat] == 0, "churned"].mean()
        print(f"  {feat:25s}: churn WITH={churn_with:.1%}, WITHOUT={churn_without:.1%}")
    else:
        q75 = accounts[feat].quantile(0.75)
        q25 = accounts[feat].quantile(0.25)
        churn_top = accounts.loc[accounts[feat] >= q75, "churned"].mean()
        churn_bot = accounts.loc[accounts[feat] <= q25, "churned"].mean()
        print(f"  {feat:25s}: churn TOP25%={churn_top:.1%}, BOT25%={churn_bot:.1%}")

output_path = "synthetic_churn_ground_truth.parquet"
accounts.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")
print(f"\nRun: python3 engine/scoring_engine.py --n 2000 --metric auc_roc_churn --data {output_path}")
