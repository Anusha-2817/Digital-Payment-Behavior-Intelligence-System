import pandas as pd

from fastapi import APIRouter
from app.schemas.transaction_schema import Transaction
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
def predict(data: list[Transaction]):

    df = pd.DataFrame(
        [item.dict() for item in data]
    )

    features = process_transactions(df)

    predictions = predict_behavior(features)

    risk_results = compute_risk(
        features,
        predictions
    )

    return {
        "results": risk_results
    }