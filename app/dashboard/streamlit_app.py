import pandas as pd
import streamlit as st

from app.services.feature_service import (
    process_transactions
)

from app.services.inference_service import (
    predict_behavior
)

from app.services.risk_service import (
    compute_risk
)

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Behavioral Intelligence Dashboard",
    layout="wide"
)

st.title("Behavioral Intelligence Dashboard")

st.markdown(
    "Upload transaction data for behavioral "
    "profiling and risk analysis."
)

# ---------------------------------
# FILE UPLOAD
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Transaction Data")

    st.dataframe(df.head())

    # ---------------------------------
    # RUN ANALYSIS
    # ---------------------------------

    if st.button("Run Behavioral Analysis"):

        # feature engineering
        features = process_transactions(df)

        # predictions
        predictions = predict_behavior(features)

        # risk scoring
        risk_results = compute_risk(
            features,
            predictions
        )

        # convert to dataframe
        results_df = pd.DataFrame(risk_results)

        # ---------------------------------
        # DISPLAY RESULTS
        # ---------------------------------

        st.subheader("Behavioral Risk Results")

        st.dataframe(results_df)

        # ---------------------------------
        # METRICS
        # ---------------------------------

        high_risk_count = (
            results_df["priority"] == "HIGH"
        ).sum()

        medium_risk_count = (
            results_df["priority"] == "MEDIUM"
        ).sum()

        low_risk_count = (
            results_df["priority"] == "LOW"
        ).sum()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "HIGH RISK USERS",
            high_risk_count
        )

        col2.metric(
            "MEDIUM RISK USERS",
            medium_risk_count
        )

        col3.metric(
            "LOW RISK USERS",
            low_risk_count
        )

        # ---------------------------------
        # RISK DISTRIBUTION
        # ---------------------------------

        st.subheader("Risk Score Distribution")

        st.bar_chart(
            results_df["risk_score"]
        )