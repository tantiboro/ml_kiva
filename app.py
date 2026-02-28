import streamlit as st
import pandas as pd
from src.ml_kiva.utils import load_data, load_model, get_shap_explainer
from src.ml_kiva.components import render_exploration, render_shap_plot

st.set_page_config(page_title="Kiva NLP Dashboard", layout="wide")

# Load Assets
df = load_data()
pipeline = load_model()

# Sidebar
st.sidebar.title("Kiva NLP")
page = st.sidebar.selectbox("Go to", ["Exploration", "Predictor"])

if page == "Exploration":
    render_exploration(df)

elif page == "Predictor":
    st.header("🤖 Funding Predictor")
    if pipeline is None:
        st.error("Model missing.")
    else:
        # User inputs
        loan_amt = st.number_input("Amount", value=500)
        sector = st.selectbox("Sector", df["sector_name"].unique())
        description = st.text_area("Borrower's Story")
        
        if st.button("Predict"):
            input_df = pd.DataFrame({
                "description_translated": [description], "loan_amount": [loan_amt],
                "lender_term": [12], "sector_name": [sector], "activity_name": ["General"]
            })
            
            proba = pipeline.predict_proba(input_df)[0][1]
            st.metric("Funding Probability", f"{proba:.2%}")
            
            render_shap_plot(pipeline, input_df, df, get_shap_explainer)
