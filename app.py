
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ML Analytics Platform",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# 2. NAVIGATION
# ==========================================

if "page" not in st.session_state:
    st.session_state.page = "Home"


def navigate(page):
    st.session_state.page = page
    st.rerun()


# ==========================================
# 3. HOME DASHBOARD
# ==========================================

def home_page():

    st.title("🤖 ML Analytics Platform")

    st.write(
        "Welcome to the Machine Learning Analytics Platform. "
        "Select a project to explore its prediction dashboard."
    )

    st.divider()

    st.subheader("Available Projects")

    col1, col2 = st.columns(2)

    # Revenue Prediction Project

    with col1:

        with st.container(border=True):

            st.markdown("## 📊 Revenue Prediction")

            st.write(
                "Predict expected campaign revenue based on "
                "campaign budget, duration, audience size, "
                "marketing channel and discount."
            )

            st.markdown("**Model:** Machine Learning Regression")

            st.markdown("**Application:** Campaign Revenue Prediction")

            if st.button(
                "Open Revenue Dashboard →",
                key="revenue_navigation",
                type="primary",
                use_container_width=True
            ):
                navigate("Revenue Prediction")

    # Lead Conversion Project

    with col2:

        with st.container(border=True):

            st.markdown("## 📈 Lead Conversion Prediction")

            st.write(
                "Estimate the probability of lead conversion "
                "within 30 days using historical lead engagement "
                "and marketing information."
            )

            st.markdown("**Model:** Logistic Regression")

            st.markdown("**Application:** Lead Conversion Prediction")

            if st.button(
                "Open Lead Dashboard →",
                key="lead_navigation",
                type="primary",
                use_container_width=True
            ):
                navigate("Lead Conversion")

    st.divider()

    st.subheader("Platform Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Projects", "02")
    col2.metric("Prediction Applications", "02")
    col3.metric("ML Platform", "Active")

    st.divider()

    st.caption(
        "Unified Machine Learning Analytics Platform | "
        "Revenue Prediction & Lead Conversion"
    )


# ==========================================
# 4. REVENUE PREDICTION DASHBOARD
# ==========================================

def revenue_prediction_page():

    if st.button("← Back to Home", key="revenue_home"):
        navigate("Home")

    st.title("📊 Campaign Revenue Predictor")

    st.write(
        "Predict expected campaign revenue using Machine Learning."
    )

    st.divider()

    # Load Revenue Model

    @st.cache_resource
    def load_revenue_model():

        artifact = joblib.load(
            "models/campaign_revenue_model.joblib"
        )

        model = artifact["model"]
        numeric_imputer = artifact["numeric_imputer"]

        return model, numeric_imputer, artifact

    try:

        loaded_model, loaded_imputer, loaded_artifact = (
            load_revenue_model()
        )

    except Exception as e:

        st.error(f"Model loading error: {e}")
        st.stop()

    # Campaign Inputs

    st.subheader("Enter Campaign Details")

    col1, col2 = st.columns(2)

    with col1:

        campaign_start_date = st.date_input(
            "Campaign Start Date",
            value=date.today()
        )

        channel = st.selectbox(
            "Marketing Channel",
            options=[
                "Search",
                "Social",
                "Email",
                "Display",
                "Affiliate"
            ]
        )

        planned_budget_inr = st.number_input(
            "Planned Budget (₹)",
            min_value=0,
            value=250000,
            step=10000
        )

    with col2:

        planned_duration_days = st.number_input(
            "Campaign Duration (Days)",
            min_value=1,
            value=30
        )

        planned_audience_size = st.number_input(
            "Planned Audience Size",
            min_value=1,
            value=150000,
            step=10000
        )

        offer_discount_pct = st.number_input(
            "Offer Discount (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0
        )

    st.divider()

    # Prediction

    if st.button(
        "🚀 Predict Campaign Revenue",
        use_container_width=True
    ):

        new_campaign = pd.DataFrame([{
            "campaign_start_date": campaign_start_date,
            "channel": channel,
            "planned_budget_inr": planned_budget_inr,
            "planned_duration_days": planned_duration_days,
            "planned_audience_size": planned_audience_size,
            "offer_discount_pct": offer_discount_pct
        }])

        # Date Feature Extraction

        new_campaign["campaign_start_date"] = pd.to_datetime(
            new_campaign["campaign_start_date"],
            errors="coerce"
        )

        new_campaign["campaign_year"] = (
            new_campaign["campaign_start_date"].dt.year
        )

        new_campaign["campaign_month"] = (
            new_campaign["campaign_start_date"]
            .dt.month.astype(str).str.zfill(2)
        )

        # Handle Missing Channel

        new_campaign["channel"] = new_campaign["channel"].fillna(
            loaded_artifact["channel_mode"]
        )

        # Numerical Preprocessing

        new_numeric = pd.DataFrame(
            loaded_imputer.transform(
                new_campaign[loaded_artifact["numeric_columns"]]
            ),
            columns=loaded_artifact["numeric_columns"],
            index=new_campaign.index
        )

        # Categorical Preprocessing

        new_categorical = pd.get_dummies(
            new_campaign[loaded_artifact["categorical_columns"]],
            dtype=float
        )

        new_categorical = new_categorical.reindex(
            columns=loaded_artifact["dummy_columns"],
            fill_value=0
        )

        # Combine Features

        new_features = pd.concat(
            [new_numeric, new_categorical],
            axis=1
        )

        # Predict Revenue

        predicted_revenue = loaded_model.predict(
            new_features
        )[0]

        # Display Result

        st.success("Prediction completed successfully!")

        st.subheader("Predicted Campaign Revenue")

        st.metric(
            label="Estimated Revenue",
            value=f"₹{predicted_revenue:,.2f}"
        )

        # Campaign Summary

        st.subheader("Campaign Summary")

        results = pd.DataFrame({
            "Channel": [channel],
            "Campaign Budget": [planned_budget_inr],
            "Duration (Days)": [planned_duration_days],
            "Audience Size": [planned_audience_size],
            "Discount (%)": [offer_discount_pct],
            "Predicted Revenue": [round(predicted_revenue, 2)]
        })

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# 5. LEAD CONVERSION DASHBOARD
# ==========================================

def lead_conversion_page():

    if st.button("← Back to Home", key="lead_home"):
        navigate("Home")

    st.title("📈 Lead Conversion Prediction")

    st.write(
        "Estimate the probability that a lead converts within 30 days."
    )

    st.divider()

    # Load Lead Conversion Model

    @st.cache_resource
    def load_lead_artifact():

        return joblib.load(
            "models/lead_conversion_model.joblib"
        )

    try:

        artifact = load_lead_artifact()

    except FileNotFoundError:

        st.error(
            "Lead conversion model file not found. "
            "Run the 20-cell notebook first."
        )
        st.stop()

    model = artifact["model"]
    numeric_imputer = artifact["numeric_imputer"]
    scaler = artifact["scaler"]
    categorical_imputer = artifact["categorical_imputer"]
    numeric_columns = artifact["numeric_columns"]
    categorical_columns = artifact["categorical_columns"]
    dummy_columns = artifact["dummy_columns"]
    threshold = artifact["threshold"]

    # Input Preparation

    def prepare_input(input_df):

        input_df = input_df.copy()

        numeric_values = numeric_imputer.transform(
            input_df[numeric_columns]
        )

        numeric_values = scaler.transform(numeric_values)

        categorical_values = categorical_imputer.transform(
            input_df[categorical_columns]
        )

        categorical_df = pd.DataFrame(
            categorical_values,
            columns=categorical_columns
        )

        categorical_df = pd.get_dummies(
            categorical_df,
            columns=categorical_columns,
            dtype=int
        )

        categorical_df = categorical_df.reindex(
            columns=dummy_columns,
            fill_value=0
        )

        return np.concatenate(
            [
                numeric_values,
                categorical_df.to_numpy()
            ],
            axis=1
        )

    # Single and Bulk Prediction Tabs

    tab_single, tab_bulk = st.tabs(
        ["Single Lead", "Bulk CSV Prediction"]
    )

    with tab_single:

        st.subheader("Enter Lead Details")

        website_visits = st.number_input(
            "Prior website visits",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

        email_engagement = st.number_input(
            "Prior email engagement count",
            min_value=0,
            max_value=100,
            value=3,
            step=1
        )

        channel = st.selectbox(
            "Channel",
            [
                "Website",
                "Email",
                "Facebook",
                "Google Ads",
                "Instagram",
                "Referral"
            ]
        )

        product = st.selectbox(
            "Product interest",
            [
                "Product A",
                "Product B",
                "Product C",
                "Product D"
            ]
        )

        if st.button(
            "Predict Conversion",
            type="primary",
            key="single_lead_prediction"
        ):

            one_lead = pd.DataFrame([{
                "prior_website_visits": website_visits,
                "prior_email_engagement_count": email_engagement,
                "channel": channel,
                "product_interest": product
            }])

            prepared = prepare_input(one_lead)

            probability = float(
                model.predict_proba(prepared)[0, 1]
            )

            prediction = int(probability >= threshold)

            st.metric(
                "Conversion Probability",
                f"{probability:.1%}"
            )

            if prediction == 1:

                st.success(
                    "High-priority lead — include in the follow-up group."
                )

            else:

                st.info(
                    "Lower-priority lead — not selected by the "
                    "current follow-up threshold."
                )

    with tab_bulk:

        st.subheader("Upload Leads CSV")

        st.write(
            "Required columns: prior_website_visits, "
            "prior_email_engagement_count, channel, product_interest."
        )

        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            key="lead_bulk_upload"
        )

        if uploaded_file is not None:

            bulk_df = pd.read_csv(uploaded_file)

            required_columns = (
                numeric_columns + categorical_columns
            )

            missing_columns = [
                col for col in required_columns
                if col not in bulk_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            else:

                prepared = prepare_input(
                    bulk_df[required_columns]
                )

                probabilities = model.predict_proba(
                    prepared
                )[:, 1]

                results = bulk_df.copy()

                results["conversion_probability"] = probabilities

                results["predicted_converted_30d"] = (
                    probabilities >= threshold
                ).astype(int)

                st.dataframe(
                    results.head(100),
                    use_container_width=True
                )

                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Predictions CSV",
                    data=csv_data,
                    file_name="lead_conversion_predictions.csv",
                    mime="text/csv"
                )

    st.divider()

    st.caption(
        f"Current follow-up threshold: {threshold:.3f}. "
        "The threshold was selected on validation data."
    )


# ==========================================
# 6. APPLICATION ROUTING
# ==========================================

if st.session_state.page == "Home":

    home_page()

elif st.session_state.page == "Revenue Prediction":

    revenue_prediction_page()

elif st.session_state.page == "Lead Conversion":

    lead_conversion_page()
