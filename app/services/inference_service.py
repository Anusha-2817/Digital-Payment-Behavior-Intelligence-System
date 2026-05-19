import joblib

from app.config import MODEL_FEATURES
# load saved artifacts
model = joblib.load("models/saved/classifier.pkl")

scaler = joblib.load("models/saved/scaler.pkl")

label_encoder = joblib.load(
    "models/saved/label_encoder.pkl"
)


def predict_behavior(features_df):

    # remove non-feature columns
    X = features_df[MODEL_FEATURES]
    # scale features
    X_scaled = scaler.transform(X)

    # predictions
    predictions = model.predict(X_scaled)

    # confidence scores
    probabilities = model.predict_proba(X_scaled)

    confidence_scores = probabilities.max(axis=1)

    # decode labels
    decoded_predictions = (
        label_encoder.inverse_transform(predictions)
    )

    results = []

    for i in range(len(features_df)):

        results.append({
            "user_id": features_df.iloc[i]["user_id"],
            "prediction": decoded_predictions[i],
            "confidence": round(
                float(confidence_scores[i]),
                3
            )
        })

    return results