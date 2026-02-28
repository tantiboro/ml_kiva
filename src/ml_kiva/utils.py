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

def setup_kaggle():
    if "KAGGLE_USERNAME" in st.secrets:
        os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
        os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        setup_kaggle()
        dataset_url = "https://www.kaggle.com/datasets/tantiboroouattara/ml-kiva-processed"
        od.download(dataset_url, data_dir='./data', force=True)
    
    df = pd.read_csv(DATA_PATH).sample(min(10000, 150000))
    df.columns = df.columns.str.lower()
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@st.cache_resource
def get_shap_explainer(_model, _background_data):
    return shap.TreeExplainer(_model, _background_data)