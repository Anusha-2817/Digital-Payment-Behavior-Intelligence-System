import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# -----------------------------
# Data Preparation
# -----------------------------
def prepare_data(df, target_column):
    # problematic stuff
    df = df.drop(columns=["user_id"])
    df["is_night"] = (df["hour_of_day"] >= 22).astype(int)
    df["is_morning"] = (df["hour_of_day"] <= 10).astype(int)
    # Drop missing target
    df = df.dropna(subset=[target_column])

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y


# -----------------------------
# Train-Test Split
# -----------------------------
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)


# -----------------------------
# Train Models
# -----------------------------
def train_models(X_train, y_train):

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoost": GradientBoostingRegressor(random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


# -----------------------------
# Evaluate Models
# -----------------------------
def evaluate_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():

        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

        results[name] = {
            "R2": round(r2, 3),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        }

    return results


# -----------------------------
# Get Best Model
# -----------------------------
def get_best_model(results):
    best_model = max(results, key=lambda x: results[x]["R2"])
    return best_model, results[best_model]


# -----------------------------
# Detect Dataset Type
# -----------------------------
def detect_dataset_type(results):
    lr_score = results.get("LinearRegression", {}).get("R2", 0)
    best_score = max([results[m]["R2"] for m in results])

    if best_score - lr_score > 0.1:
        return "Non-linear"
    else:
        return "Linear"