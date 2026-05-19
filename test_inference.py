import pandas as pd

from app.services.inference_service import (
    predict_behavior
)

from app.services.feature_service import (
    process_transactions
)
from app.services.risk_service import (
    compute_risk
)

# load raw transactions
df = pd.read_csv(
    "data/raw/upi_transactions_behavioral.csv"
)

# generate features
features = process_transactions(df)

# run inference
results = predict_behavior(features)
risk_results = compute_risk(
    features,
    results
)

print(results[:5])
print(risk_results[:5])
print(features[[
    "night_transaction_ratio",
    "failed_txn_spike_ratio",
    "high_value_txn_ratio",
    "merchant_diversity"
]].head())