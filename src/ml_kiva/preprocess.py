import pandas as pd
import re
import spacy
import unicodedata
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model for lemmatization
# Note: Ensure you run 'python -m spacy download en_core_web_sm' in your conda env
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean_text(text):
    """Clean text by stripping HTML, removing accents, and lemmatizing."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    # 1. Strip HTML
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Normalize Accents (convert Bogotá to Bogota)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # 3. Process with spaCy
    doc = nlp(text.lower())
    
    # Keep only alphabetic tokens, excluding stop words
    # Note: Using .lemma_ for root words
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
    return " ".join(lemmas)

def load_and_clean_data(data_path):
    """Loads the balanced dataset and applies linguistic cleaning."""
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()

    # Define column types
    text_col = "description_translated"
    num_cols = ["loan_amount", "lender_term"]
    cat_cols = ["sector_name", "activity_name"]

    # Fill NAs and apply cleaning
    df[text_col] = df[text_col].fillna("").apply(clean_text) # CRITICAL: Calls clean_text
    
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")

    X = df[[text_col] + num_cols + cat_cols]
    # Standardize target mapping
    y = (df["status"].str.lower() == "funded").astype(int)

    return X, y

def get_preprocessor(text_col, num_cols, cat_cols):
    """Returns the ColumnTransformer for the pipeline."""
    return ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(max_features=2000, stop_words="english"),
                text_col,
            ),
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]  
        )