def compute_risk(features_df, predictions):

    risk_results = []

    for i in range(len(features_df)):

        row = features_df.iloc[i]

        prediction = predictions[i]

        risk_score = 0

        reasons = []

        # ---------------------------------
        # NIGHT SPENDING RISK
        # ---------------------------------

        if row["night_transaction_ratio"] > 0.50:

            risk_score += 20

            reasons.append(
                "High late-night transaction activity"
            )

        # ---------------------------------
        # FAILED TRANSACTION RISK
        # ---------------------------------

        if row["failed_txn_spike_ratio"] > 0.30:

            risk_score += 25

            reasons.append(
                "Spike in failed transaction attempts"
            )

        # ---------------------------------
        # HIGH VALUE SPENDING
        # ---------------------------------

        if row["high_value_txn_ratio"] > 0.40:

            risk_score += 20

            reasons.append(
                "Frequent high-value transactions"
            )

        # ---------------------------------
        # MERCHANT DIVERSITY
        # ---------------------------------

        if row["merchant_diversity"] > 12:

            risk_score += 15

            reasons.append(
                "Unusually high merchant diversity"
            )

        # ---------------------------------
        # LOW MODEL CONFIDENCE
        # ---------------------------------

        if prediction["confidence"] < 0.50:

            risk_score += 10

            reasons.append(
                "Low behavioral prediction confidence"
            )

        # ---------------------------------
        # RISK LEVELS
        # ---------------------------------

        if risk_score >= 70:
            priority = "HIGH"

        elif risk_score >= 40:
            priority = "MEDIUM"

        else:
            priority = "LOW"

        risk_results.append({

            "user_id": prediction["user_id"],

            "predicted_behavior": (
                prediction["prediction"]
            ),

            "confidence": (
                prediction["confidence"]
            ),

            "risk_score": risk_score,

            "priority": priority,

            "reasons": reasons
        })

    return risk_results