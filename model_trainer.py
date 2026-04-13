import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("upi_transactions_messy.csv")

# -----------------------------
# Drop rows with missing target
# -----------------------------
df = df.dropna(subset=["transaction_amount"])

# -----------------------------
# Encode categorical columns
# -----------------------------
categorical_cols = df.select_dtypes(include="object").columns

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# Define features and target
# -----------------------------
X = df.drop("transaction_amount", axis=1)
y = df["transaction_amount"]

# -----------------------------
# Train test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "GradientBoost": GradientBoostingRegressor()
}

results = {}

print("\n====== MODEL TRAINING RESULTS ======\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results[name] = {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    }

    print(name)
    print("R2   :", round(r2, 3))
    print("MAE  :", round(mae, 2))
    print("RMSE :", round(rmse, 2))
    print("-----------------------")

# -----------------------------
# Best model
# -----------------------------
best_model = max(results, key=lambda x: results[x]["R2"])

print("\nBest Model:", best_model)