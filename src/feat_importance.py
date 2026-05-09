import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Extract Feature Importance
# -----------------------------
def get_feat_importance(model, feature_names):
    
    # Tree-based models only
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not support feature importance")

    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(by="Importance", ascending=False)

    return feature_df


# -----------------------------
# Print Top Features
# -----------------------------
def print_top_features(feature_df, top_n=10):

    print("\n====== TOP FEATURES ======\n")

    for i, row in feature_df.head(top_n).iterrows():
        print(f"{row['Feature']} : {round(row['Importance'], 3)}")


# -----------------------------
# Plot Feature Importance
# -----------------------------
def plot_feat_importance(feature_df, top_n=10):

    top_features = feature_df.head(top_n)

    plt.figure()
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.gca().invert_yaxis()

    plt.title("Top Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    plt.tight_layout()
    plt.show()