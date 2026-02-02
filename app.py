import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from pathlib import Path

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="Kiva NLP Dashboard", layout="wide")

# Use relative paths consistent with your VS Code structure
DATA_PATH = Path("data/processed/processed_loans_balanced.csv")
MODEL_PATH = Path("models/kiva_pipeline.joblib")


@st.cache_resource
def load_model():
    """Load the trained pipeline."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def get_exploration_data():
    """Load a sample of the dataset for visualization."""
    if DATA_PATH.exists():
        # Load sample for performance
        df = pd.read_csv(DATA_PATH).sample(min(10000, 180000))
        df.columns = df.columns.str.lower()
        return df
    return pd.DataFrame()


# Global assignments
pipeline = load_model()
df_explore = get_exploration_data()

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("Kiva NLP Dashboard")
page = st.sidebar.selectbox("Go to", ["Data Exploration", "Loan Funding Predictor"])

# --- PAGE 1: DATA EXPLORATION ---
if page == "Data Exploration":
    st.header("📊 Loan Data Insights")
    if not df_explore.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(
                df_explore, names="status", title="Funding Status Distribution"
            )
            st.plotly_chart(fig_pie)
        with col2:
            sector_counts = df_explore["sector_name"].value_counts().head(10)
            st.bar_chart(sector_counts)

        st.subheader("Loan Amount Distribution by Sector")
        fig_box = px.box(df_explore, x="sector_name", y="loan_amount", color="status")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning(f"⚠️ Exploration data not found at: {DATA_PATH.absolute()}")

# --- PAGE 2: PREDICTOR ---
elif page == "Loan Funding Predictor":
    st.header("🤖 Multi-Column ML Predictor")

    if pipeline is None:
        st.error(
            f"❌ Model artifact not found at {MODEL_PATH.absolute()}. Please run train.py first."
        )
    else:
        # Define User Inputs BEFORE the button to avoid NameError
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            loan_amt = st.number_input("Loan Amount ($)", min_value=25, value=500)
            lender_term = st.slider("Lender Term (Months)", 1, 60, 12)
        with col_in2:
            sectors = (
                df_explore["sector_name"].unique()
                if not df_explore.empty
                else ["Agriculture", "Retail", "Food"]
            )
            sector = st.selectbox("Sector", sectors)
            activity = st.text_input("Activity", "General")

        description = st.text_area(
            "Borrower's Story (NLP Feature)",
            placeholder="Describe how the loan will be used...",
        )

        if st.button("Predict Funding Success"):
            # 1. Package inputs into DataFrame
            input_df = pd.DataFrame(
                {
                    "description_translated": [description],
                    "loan_amount": [loan_amt],
                    "lender_term": [lender_term],
                    "sector_name": [sector],
                    "activity_name": [activity],
                }
            )

            # 2. Run Inference
            proba = pipeline.predict_proba(input_df)[0][1]
            prediction = "Funded" if proba >= 0.5 else "Expired"

            # 3. Display Metrics
            st.divider()
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Probability of Funding", f"{proba:.2%}")
            with res_col2:
                st.metric("Prediction", prediction)
            st.progress(proba)

            # --- 4. DYNAMIC FEATURE IMPORTANCE ---
            st.subheader("🧐 Why this prediction?")
            prep = pipeline.named_steps["preprocessor"]
            rf = pipeline.named_steps["rf"]

            # Reconstruct names from all transformers
            tfidf_names = prep.named_transformers_["text"].get_feature_names_out()
            num_names = ["loan_amount", "lender_term"]
            cat_names = prep.named_transformers_["cat"].get_feature_names_out()
            all_feature_names = np.concatenate([tfidf_names, num_names, cat_names])

            importances = rf.feature_importances_
            importance_df = pd.DataFrame(
                {"feature": all_feature_names, "importance": importances}
            )

            # Calculate local importance for the current text
            user_words = description.lower().split()
            active_text_features = importance_df[
                importance_df["feature"].isin(user_words)
            ]
            active_other_features = importance_df[
                importance_df["feature"].isin(num_names + list(cat_names))
            ]

            local_importance = pd.concat([active_text_features, active_other_features])
            local_importance = local_importance.sort_values(
                by="importance", ascending=False
            ).head(10)

            if not local_importance.empty:
                fig_imp = px.bar(
                    local_importance,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top Factors for this Prediction",
                    color="importance",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.write("No high-impact keywords detected.")
