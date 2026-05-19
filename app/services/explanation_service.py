def generate_summary(features, risk_score):

    reasons = []

    if features["night_transaction_ratio"] > 0.6:
        reasons.append("High late-night transaction activity detected.")

    if features["merchant_diversity"] > 15:
        reasons.append("Rapid merchant diversification observed.")

    if risk_score > 70:
        status = "HIGH RISK"
    else:
        status = "MODERATE RISK"

    return {
        "status": status,
        "summary": reasons
    }