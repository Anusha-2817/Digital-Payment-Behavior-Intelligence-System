import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

# METRICS
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv(
    "data/processed/user_behavior_features.csv"
)

# -----------------------------
# FEATURES
# -----------------------------

basic_features = [
    "txn_count",
    "avg_transaction_value",
    "merchant_diversity",
    "weekend_spending_ratio",
    "night_transaction_ratio",
    "failed_transaction_rate",
    "recurring_payment_ratio"
]

behavioral_features = [

    # BASIC
    "txn_count",
    "avg_transaction_value",
    "merchant_diversity",
    "weekend_spending_ratio",
    "night_transaction_ratio",
    "failed_transaction_rate",
    "recurring_payment_ratio",

    # ADVANCED
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





# -----------------------------
# ENCODE LABELS
# -----------------------------
y = df["behavior_personality"]
print(df["behavior_personality"].value_counts())
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# -----------------------------
# MODELS
# -----------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

# -----------------------------
# TRAIN + EVALUATE
# -----------------------------

def train_and_evaluate(feature_set, experiment_name):

    print("\n" + "#"*60)
    print(f"EXPERIMENT: {experiment_name}")
    print("#"*60)

    X = df[feature_set]
    print("\nFEATURE COLUMNS:")
    print(X.columns.tolist())

    print("\nTARGET COLUMN:")
    print(y.name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    results = {}

    for name, model in models.items():

        print("\n" + "="*50)
        print(f"{name}")
        print("="*50)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        results[name] = accuracy

        print(f"\nAccuracy: {accuracy:.4f}")

    best_model = max(results, key=results.get)

    print("\n" + "-"*50)
    print("MODEL COMPARISON")
    print("-"*50)

    for model_name, score in results.items():
        print(f"{model_name}: {score:.4f}")

    print(f"\nBest Model: {best_model}")

    return results

# ---------------------------------
# EXPERIMENT 1
# BASIC FEATURES
# ---------------------------------

basic_results = train_and_evaluate(
    basic_features,
    "BASIC FEATURES"
)

# ---------------------------------
# EXPERIMENT 2
# FULL BEHAVIORAL FEATURES
# ---------------------------------

behavior_results = train_and_evaluate(
    behavioral_features,
    "FULL BEHAVIORAL FEATURES"
)