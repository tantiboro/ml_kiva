import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_clean_data(data_path):
    """Loads the balanced dataset and handles initial NA values."""
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()

    # Define column types
    text_col = "description_translated"
    num_cols = ["loan_amount", "lender_term"]
    cat_cols = ["sector_name", "activity_name"]

    # Fill NAs
    df[text_col] = df[text_col].fillna("")
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")

    X = df[[text_col] + num_cols + cat_cols]
    y = (df["status"] == "funded").astype(int)

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
