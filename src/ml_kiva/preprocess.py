import pandas as pd
import re
import unicodedata
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from ml_kiva.nlp import get_nlp 

def clean_text(text):
    """Clean text by stripping HTML, removing accents, and lemmatizing."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    # 1. Strip HTML
    text = re.sub(r"<.*?>", "", text)
    # 2. Normalize Accents
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    # 3. Process with spaCy
    nlp = get_nlp()
    doc = nlp(text.lower())
    lemmas = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]
    return " ".join(lemmas)

def load_and_clean_data(data_path):
    """Loads dataset adaptively based on the environment."""
    
    # --- ADAPTIVE LOADING ---
    # Detect if we are on the Pixelbook (usually 'penguin') vs Colab
    is_local = os.uname().nodename == 'penguin'
    
    if is_local:
        print("🧪 Local environment detected. Subsampling 10k rows...")
        df = pd.read_csv(data_path, nrows=10000)
    else:
        print("🚀 Cloud environment detected. Loading full dataset...")
        df = pd.read_csv(data_path)

    df.columns = df.columns.str.lower()
    text_col = "description_translated"
    num_cols = ["loan_amount", "lender_term"]
    cat_cols = ["sector_name", "activity_name"]

    # --- ADAPTIVE CLEANING ---
    if is_local:
        # Fast path for Pixelbook to avoid Segmentation Faults/Hangs
        print("🧹 Applying fast regex cleaning...")
        df[text_col] = df[text_col].fillna("").str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)
    else:
        # Deep path for Colab/Production
        print("🧠 Applying deep NLP lemmatization...")
        df[text_col] = df[text_col].fillna("").apply(clean_text)
    
    # Missing value handling
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")

    X = df[[text_col] + num_cols + cat_cols]
    y = (df["status"].str.lower() == "funded").astype(int)
    return X, y

def get_preprocessor(text_col, num_cols, cat_cols):
    """Returns the ColumnTransformer for the pipeline."""
    return ColumnTransformer(
        [
            ("text", TfidfVectorizer(max_features=2000, stop_words="english"), text_col),
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
