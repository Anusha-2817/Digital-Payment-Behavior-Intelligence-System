#------------discover natural behavior groups------------------
#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#load dataset
df = pd.read_csv("data/processed/user_behavior_features.csv")
#select features for clustering
features = [
    "txn_count",
    "avg_transaction_value",
    "merchant_diversity",
    "weekend_spending_ratio",
    "night_transaction_ratio",
    "failed_transaction_rate",
    "recurring_payment_ratio"
]
# scale features !! avoid domination by magnitude
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
#determine optimal clusters using elbow method
inertia_values = []

K_range = range(2, 9)

for k in K_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )
    kmeans.fit(X_scaled)

    inertia_values.append(kmeans.inertia_)
#plot elbow curve

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
# from elbow curve, choose n_clusters=3
n_clusters = 3
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)
#attach cluster labels to original dataframe
df["cluster"] = clusters
print(df["cluster"].value_counts())  #KMeans needs meaningful spread

#cluster summary
cluster_summary = df.groupby("cluster")[features].mean()
print(cluster_summary.to_string())
#save clustered dataset
df.to_csv( 
    "data/processed/clustered_users.csv",
    index=False
)

print("\nClustered dataset saved!")