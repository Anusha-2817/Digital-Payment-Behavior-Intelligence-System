#-----------convert clusters into semantic labels------------

import pandas as pd
import random

random.seed(42)

# -----------------------------
# LOAD FEATURE DATASET
# -----------------------------

df = pd.read_csv(
    "data/processed/user_behavior_features.csv"
)

# -----------------------------
# CREATE SEMANTIC LABELS
# -----------------------------

labels = []

for _, row in df.iterrows():

    score = 0

    # ---------------------------------
    # SPENDING SIGNALS
    # ---------------------------------

    if row["avg_transaction_value"] > 5000:
        score += 2

    if row["high_value_txn_ratio"] > 0.20:
        score += 2

    if row["spending_variance"] > 10000:
        score += 1

    # ---------------------------------
    # BEHAVIOR SIGNALS
    # ---------------------------------

    if row["night_transaction_ratio"] > 0.40:
        score += 1

    if row["category_switch_rate"] > 0.50:
        score += 1

    if row["weekend_night_ratio"] > 0.30:
        score += 1

    # ---------------------------------
    # CONTRADICTION PROBABILITY
    # ---------------------------------

    contradiction_probability = 0.15

    if random.random() < contradiction_probability:
        score += random.choice([-2, -1, 1, 2])

    # ---------------------------------
    # FINAL LABELS
    # ---------------------------------

    if score <= 2:
        label = "Conservative User"

    elif score <= 5:
        label = "Balanced User"

    else:
        label = "Aggressive User"

    labels.append(label)

# -----------------------------
# SAVE LABELS
# -----------------------------

df["spender_type"] = labels

df.to_csv(
    "data/processed/labeled_users.csv",
    index=False
)

print("\nSemantic labeling complete!")
print(df[["user_id", "spender_type"]].head())