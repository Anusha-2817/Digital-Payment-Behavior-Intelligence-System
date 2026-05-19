import pandas as pd

from fastapi import APIRouter

from app.services.feature_service import (
    process_transactions
)

from app.services.inference_service import (
    predict_behavior
)

from app.services.risk_service import (
    compute_risk
)

router = APIRouter()


@router.post("/predict")
def predict(data: list[dict]):

    # convert incoming JSON to dataframe
    df = pd.DataFrame(data)

    # feature engineering
    features = process_transactions(df)

    # inference
    predictions = predict_behavior(features)

    # risk scoring
    risk_results = compute_risk(
        features,
        predictions
    )

    return {
        "results": risk_results
    }