import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Prepare Data (NO TARGET)
# -----------------------------
def prepare_clustering_data(df, target_column):

    # Drop target
    df = df.drop(columns=[target_column])

    # Drop leakage + IDs
    df = df.drop(columns=["user_id", "balance_after"], errors='ignore')

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    return scaled_data, df.columns


# -----------------------------
# Run KMeans
# -----------------------------
def run_kmeans(data, n_clusters=3):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)

    return labels


# -----------------------------
# Attach clusters to dataframe
# -----------------------------
def attach_clusters(df, labels):

    df_clustered = df.copy()
    df_clustered["cluster"] = labels

    return df_clustered


# -----------------------------
# Cluster Summary
# -----------------------------
def cluster_summary(df_clustered):

    print("\n====== CLUSTER SUMMARY ======\n")

    summary = df_clustered.groupby("cluster").mean(numeric_only=True).round(2)

    print(summary)

    # -----------------------------
    # Assign meaningful labels
    # -----------------------------
    avg_spend = summary["transaction_amount"]

    sorted_clusters = avg_spend.sort_values()

    cluster_labels = {}

    cluster_labels[sorted_clusters.index[0]] = "Low Spenders"
    cluster_labels[sorted_clusters.index[1]] = "Regular Users"
    cluster_labels[sorted_clusters.index[2]] = "High Spenders"

    print("\n====== CLUSTER INTERPRETATION ======\n")

    for cluster, label in cluster_labels.items():
        print(f"Cluster {cluster} → {label}")
    return cluster_labels