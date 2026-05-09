from src import data_profiler
from src import model_trainer
from src import feat_importance
from src import clustering
# -----------------------------
# LOAD DATA
# -----------------------------
df = data_profiler.load_data("data/upi_transactions_messy.csv")

# -----------------------------
# (Optional) PROFILE DATA
# -----------------------------
print("\n========== DATASET OVERVIEW ==========")
print(data_profiler.dataset_overview(df))

# -----------------------------
# TRAINING PIPELINE
# -----------------------------
target = "transaction_amount"

X, y = model_trainer.prepare_data(df, target)

X_train, X_test, y_train, y_test = model_trainer.split_data(X, y)

models = model_trainer.train_models(X_train, y_train)

results = model_trainer.evaluate_models(models, X_test, y_test)

print("\n====== MODEL RESULTS ======\n")
for model, metrics in results.items():
    print(model, metrics)

best_model, best_metrics = model_trainer.get_best_model(results)

dataset_type = model_trainer.detect_dataset_type(results)

print("\n====== FINAL SUMMARY ======")
print("Best Model:", best_model)
print("Metrics:", best_metrics)
print("Dataset Type:", dataset_type)

# -----------------------------
# FEATURE IMPORTANCE (NEW)
# -----------------------------
best_model_obj = models[best_model]

feature_df = feat_importance.get_feat_importance(
    best_model_obj,
    X_train.columns
)

feat_importance.print_top_features(feature_df)

feat_importance.plot_feat_importance(feature_df)

from src import clustering

# -----------------------------
# CLUSTERING
# -----------------------------
cluster_data, feature_names = clustering.prepare_clustering_data(
    df,
    target_column="transaction_amount"
)

labels = clustering.run_kmeans(cluster_data, n_clusters=3)

df_clustered = clustering.attach_clusters(df, labels)

cluster_labels = clustering.cluster_summary(df_clustered)
print("\n==============================")
print(" DATASET PERSONALITY REPORT")
print("==============================\n")

print(f"Samples: {df.shape[0]}")
print(f"Features: {df.shape[1]}")

print("\nProblem Type: Regression")
print(f"Dataset Type: {dataset_type}")

print(f"\nBest Model: {best_model}")
print(f"Performance: {best_metrics}")

print("\nTop Features:")
for i, row in feature_df.head(5).iterrows():
    print(f"- {row['Feature']}")

print("\nClusters Found: 3")
for cluster, label in cluster_labels.items():
    print(f"- Cluster {cluster} → {label}")

print("\nRecommendation:")
print("Tree-based models perform better due to non-linear relationships.")