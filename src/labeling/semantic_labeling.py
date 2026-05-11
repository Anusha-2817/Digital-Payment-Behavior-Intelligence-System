#-----------convert clusters into semantic labels------------

import pandas as pd

# load clustered dataset
df = pd.read_csv(
    "data/processed/clustered_users.csv"
)

# cluster → semantic label mapping
cluster_labels = {
    0: "Conservative Frequent User",
    1: "Balanced Utility User",
    2: "Aggressive Premium User"
}

# create new column
df["spender_type"] = df["cluster"].map(cluster_labels)

# save labeled dataset
df.to_csv(
    "data/processed/labeled_users.csv",
    index=False
)

print(df[["user_id", "cluster", "spender_type"]].head().to_string())