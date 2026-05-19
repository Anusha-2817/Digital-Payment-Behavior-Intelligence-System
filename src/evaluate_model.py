import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv(
    "data/processed/user_behavior_features.csv"
)

# --------------------------------
# FEATURES
# --------------------------------

features = [

    "txn_count",
    "avg_transaction_value",
    "merchant_diversity",
    "weekend_spending_ratio",
    "night_transaction_ratio",
    "failed_transaction_rate",
    "recurring_payment_ratio",

    "spending_variance",
    "high_value_txn_ratio",
    "late_night_txn_count",
    "weekend_night_ratio",
    "category_switch_rate",
    "unique_upi_apps",
    "avg_transaction_gap",
    "transaction_gap_variance",
    "failed_txn_spike_ratio"
]

X = df[features]

y = df["behavior_personality"]

# --------------------------------
# ENCODE TARGET
# --------------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# FEATURE SCALING
# --------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------
# TRAIN MODEL
# --------------------------------

model = GradientBoostingClassifier(
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------
# PREDICTIONS
# --------------------------------

y_pred = model.predict(X_test)

# --------------------------------
# ACCURACY
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))

# --------------------------------
# CLASSIFICATION REPORT
# --------------------------------

print("\nCLASSIFICATION REPORT\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

# --------------------------------
# CONFUSION MATRIX
# --------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.title("Behavior Personality Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("src/saved outputs/plots/confusion_matrix.png")
plt.show()

# --------------------------------
# FEATURE IMPORTANCE
# --------------------------------

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE\n")

print(importance_df)

# --------------------------------
# FEATURE IMPORTANCE PLOT
# --------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.tight_layout()
plt.savefig("src/outputs/plots/feature_importance.png")
plt.show()