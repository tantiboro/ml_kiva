import streamlit as st
import pandas as pd
import joblib
import os
import opendatasets as od
import shap
from pathlib import Path

# Constants
DATA_PATH = Path("./data/ml-kiva-processed/processed_loans_balanced.csv")
MODEL_PATH = Path("models/kiva_pipeline.joblib")

import json

def setup_kaggle():
    if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
        # Create a dictionary with credentials
        kaggle_data = {
            "username": st.secrets["KAGGLE_USERNAME"],
            "key": st.secrets["KAGGLE_KEY"]
        }
        
        # Write the kaggle.json file to the current directory
        # opendatasets looks for this file by default
        with open("kaggle.json", "w") as f:
            json.dump(kaggle_data, f)
        
        # Also set environment variables as a backup
        os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
        os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        setup_kaggle() # Now creates the physical .json file
        dataset_url = "https://www.kaggle.com/datasets/tantiboroouattara/ml-kiva-processed"
        
        # Use force=False to avoid redundant downloads
        od.download(dataset_url, data_dir='./data', force=False)
    
    # Check if file exists after download attempt
    if not DATA_PATH.exists():
        st.error(f"❌ Data file not found at {DATA_PATH}. Check directory structure.")
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
    # Adaptive sampling for cloud performance
    sample_size = min(10000, len(df))
    df = df.sample(sample_size, random_state=42)
    df.columns = df.columns.str.lower()
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@st.cache_resource
def get_shap_explainer(_model, _background_data):
    return shap.TreeExplainer(_model, _background_data)