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
    "data/processed/labeled_users.csv"
)

# -----------------------------
# FEATURES
# -----------------------------

features = [
    "txn_count",
    "avg_transaction_value",
    "merchant_diversity",
    "weekend_spending_ratio",
    "night_transaction_ratio",
    "failed_transaction_rate",
    "recurring_payment_ratio"
]

X = df[features]

y = df["spender_type"]

# -----------------------------
# ENCODE LABELS
# -----------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

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

results = {}

for name, model in models.items():

    print("\n" + "="*50)
    print(f"{name}")
    print("="*50)

    # train
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"\nAccuracy: {accuracy:.4f}")

    # classification report
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    # confusion matrix
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

# -----------------------------
# BEST MODEL
# -----------------------------

best_model = max(results, key=results.get)

print("\n" + "="*50)
print("FINAL MODEL COMPARISON")
print("="*50)

for model_name, score in results.items():
    print(f"{model_name}: {score:.4f}")

print(f"\nBest Model: {best_model}")