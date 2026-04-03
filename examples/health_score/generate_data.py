"""Generate synthetic SaaS account data with planted signals for validation.

Creates 5,000 accounts with 15 features and 3 outcomes.
The signals are KNOWN - so we can verify the engine recovers them.

Planted signals:
  1. alert_created + session_depth → strong retention predictor
  2. activation_speed (fast = better) → strong retention + LTV predictor
  3. credit_limit_hits → POSITIVE engagement signal (counterintuitive)
  4. noise_feature_1, noise_feature_2 → no predictive power (should get low weight)
  5. mrr → correlated with LTV but NOT with retention (guardrail test)

Run: python3 generate_synthetic.py
Output: synthetic_ground_truth.parquet
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000

# --- Base features (all 0-1 normalized) ---
accounts = pd.DataFrame({
    "account_id": [f"acct_{i:05d}" for i in range(N)],

    # Behavioral features (from product analytics)
    "search_intensity": np.clip(np.random.exponential(0.3, N), 0, 1),
    "session_depth": np.clip(np.random.exponential(0.25, N), 0, 1),
    "days_active_norm": np.clip(np.random.exponential(0.2, N), 0, 1),
    "export_activity": np.clip(np.random.exponential(0.15, N), 0, 1),

    # Milestone features (binary)
    "has_searched": np.random.binomial(1, 0.6, N).astype(float),
    "has_saved": np.random.binomial(1, 0.3, N).astype(float),
    "alert_created": np.random.binomial(1, 0.15, N).astype(float),
    "has_exported": np.random.binomial(1, 0.25, N).astype(float),

    # Speed-to-activation (0-1, higher = faster = better)
    "activation_speed": np.clip(np.random.beta(2, 5, N), 0, 1),

    # Frustration signals
    "credit_limit_hits": np.clip(np.random.exponential(0.1, N), 0, 1),
    "rage_clicks": np.clip(np.random.exponential(0.08, N), 0, 1),

    # Context
    "email_engagement": np.clip(np.random.normal(0.3, 0.15, N), 0, 1),
    "mrr": np.clip(np.random.lognormal(5.0, 0.8, N), 50, 5000),

    # NOISE - these should get zero or near-zero weight
    "noise_feature_1": np.random.uniform(0, 1, N),
    "noise_feature_2": np.random.uniform(0, 1, N),
})

# --- Generate outcomes based on KNOWN signal relationships ---

# Retention probability: driven by alert_created, session_depth, activation_speed
# NOT driven by MRR, noise features
retention_logit = (
    2.5 * accounts["alert_created"]       # Strongest signal (binary)
    + 2.0 * accounts["session_depth"]      # Second strongest
    + 1.8 * accounts["activation_speed"]   # Third
    + 1.0 * accounts["has_exported"]       # Moderate
    + 0.8 * accounts["credit_limit_hits"]  # Positive! (counterintuitive)
    + 0.5 * accounts["days_active_norm"]   # Weak positive
    + 0.3 * accounts["email_engagement"]   # Very weak
    - 0.5 * accounts["rage_clicks"]        # Slightly negative for retention (different from engagement)
    + 0.0 * accounts["mrr"]               # NO effect on retention
    + 0.0 * accounts["noise_feature_1"]    # NO effect
    + 0.0 * accounts["noise_feature_2"]    # NO effect
    - 1.5                                  # Intercept (base churn rate)
)
retention_prob = 1 / (1 + np.exp(-retention_logit))
accounts["retained_6mo"] = np.random.binomial(1, retention_prob, N).astype(float)
accounts["churned"] = 1.0 - accounts["retained_6mo"]

# LTV: driven by MRR * tenure, where tenure correlates with retention signals
# MRR matters for LTV (not retention), plus behavioral depth
base_tenure = np.where(accounts["retained_6mo"] == 1,
                       np.clip(np.random.lognormal(2.5, 0.5, N), 3, 60),
                       np.clip(np.random.lognormal(1.5, 0.8, N), 1, 24))
accounts["tenure_months"] = base_tenure
accounts["actual_ltv"] = accounts["mrr"] * accounts["tenure_months"] * (
    1 + 0.3 * accounts["alert_created"]
    + 0.2 * accounts["activation_speed"]
    + 0.1 * accounts["session_depth"]
)

# Summary stats
print(f"Generated {N} synthetic accounts")
print(f"\nRetention rate: {accounts['retained_6mo'].mean():.1%}")
print(f"Churn rate: {accounts['churned'].mean():.1%}")
print(f"Mean LTV: ${accounts['actual_ltv'].mean():,.0f}")
print(f"Median MRR: ${accounts['mrr'].median():,.0f}")

print(f"\n--- Signal verification ---")
for feat in ["alert_created", "session_depth", "activation_speed",
             "credit_limit_hits", "mrr", "noise_feature_1"]:
    if accounts[feat].nunique() <= 2:
        # Binary feature
        ret_with = accounts.loc[accounts[feat] > 0, "retained_6mo"].mean()
        ret_without = accounts.loc[accounts[feat] == 0, "retained_6mo"].mean()
        print(f"  {feat:25s}: retention WITH={ret_with:.1%}, WITHOUT={ret_without:.1%}")
    else:
        # Continuous: compare top vs bottom quartile
        q75 = accounts[feat].quantile(0.75)
        q25 = accounts[feat].quantile(0.25)
        ret_top = accounts.loc[accounts[feat] >= q75, "retained_6mo"].mean()
        ret_bot = accounts.loc[accounts[feat] <= q25, "retained_6mo"].mean()
        print(f"  {feat:25s}: retention TOP25%={ret_top:.1%}, BOT25%={ret_bot:.1%}")

# Save
output_path = "synthetic_ground_truth.parquet"
accounts.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")
