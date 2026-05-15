import pandas as pd

# load transaction dataset
df = pd.read_csv("data/raw/upi_transactions_behavioral.csv")
print(df.columns)

# store engineered rows
feature_rows = []

# group by user
grouped = df.groupby("user_id")

for user_id, group in grouped:

    row = {}

    row["user_id"] = user_id

    # -----------------------------
    # SPENDING FEATURES
    # -----------------------------

    row["txn_count"] = len(group)

    row["avg_transaction_value"] = group["transaction_amount"].mean()

    row["max_transaction"] = group["transaction_amount"].max()

    row["total_spend"] = group["transaction_amount"].sum()

    row["spending_variance"] = (
    group["transaction_amount"].std()
    )

    row["high_value_txn_ratio"] = (
        (group["transaction_amount"] > 10000)
    ).mean()
    

    # -----------------------------
    # TIME FEATURES
    # -----------------------------

    row["weekend_spending_ratio"] = group["is_weekend"].mean()

    row["night_transaction_ratio"] = (
        group["hour_of_day"] >= 22
    ).mean()
    row["late_night_txn_count"] = (
        (group["hour_of_day"] >= 22)
    ).sum()

    row["weekend_night_ratio"] = (
        (
            (group["is_weekend"] == 1) &
            (group["hour_of_day"] >= 22)
        ).mean()
    )
    # -----------------------------
    # MERCHANT FEATURES
    # -----------------------------

    row["favorite_category"] = (
        group["merchant_category"]
        .mode()[0]
    )

    row["merchant_diversity"] = (
        group["merchant_category"]
        .nunique()
    )
    category_changes = (
        group["merchant_category"]
        != group["merchant_category"].shift()
    ).sum()

    row["category_switch_rate"] = (
        category_changes / len(group)
    )

    # -----------------------------
    # DIGITAL FEATURES
    # -----------------------------

    row["preferred_payment_method"] = (
        group["payment_method"]
        .mode()[0]
    )

    row["avg_network_latency"] = (
        group["network_latency"]
        .mean()
    )
    row["unique_upi_apps"] = (
        group["upi_app"]
        .nunique()
    )
    # -----------------------------
    # RISK FEATURES
    # -----------------------------

    row["failed_transaction_rate"] = (
        group["failed_transaction_attempts"]
        .mean()
    )
    row["avg_transaction_gap"] = (
        group["transaction_gap_minutes"]
        .mean()
    )

    row["transaction_gap_variance"] = (
        group["transaction_gap_minutes"]
        .std()
    )
    row["failed_txn_spike_ratio"] = (
        (group["failed_transaction_attempts"] >= 3)
    ).mean()

    row["recurring_payment_ratio"] = (
        group["is_recurring"]
        .mean()
    )

    # -----------------------------
    # USER METADATA
    # -----------------------------

    row["income_level"] = (
        group["income_level"]
        .iloc[0]
    )

    row["user_profile"] = (
        group["user_profile"]
        .iloc[0]
    )
    row["behavior_personality"] = (
    group["behavior_personality"]
    .iloc[0]
    )

    feature_rows.append(row)

# final dataframe
features_df = pd.DataFrame(feature_rows)

# save
features_df.to_csv(
    "data/processed/user_behavior_features.csv",
    index=False
)

print("Feature engineering complete!")
print(features_df.head())
print("\nShape:", features_df.shape)