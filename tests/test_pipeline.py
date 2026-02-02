import pytest
import pandas as pd
import joblib
from pathlib import Path
from ml_kiva.preprocess import load_and_clean_data

# Paths for local testing (consistent with your VS Code sidebar)
MODEL_PATH = Path("models/kiva_pipeline.joblib")


def test_preprocessing_logic(tmp_path):
    """
    Unit test for the spaCy cleaning logic in preprocess.py.
    Ensures HTML is stripped and columns are standard lowercase.
    """
    # Create a temporary mock CSV
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test_data.csv"

    mock_df = pd.DataFrame(
        {
            "DESCRIPTION_TRANSLATED": ["<p>Funding for <b>seeds</b>.</p>"],
            "LOAN_AMOUNT": [500],
            "STATUS": ["funded"],
            "ACTIVITY_NAME": ["Farming"],
            "SECTOR_NAME": ["Agriculture"],
            "LENDER_TERM": [12],
            "SECTOR_ID": [1],
            "ACTIVITY_ID": [10],
        }
    )
    mock_df.to_csv(p, index=False)

    # Run the preprocessor
    X, y = load_and_clean_data(p)

    # Assertions
    assert "<p>" not in X["description_translated"].iloc[0]  # HTML Stripped
    assert all(col.islower() for col in X.columns)  # Standardized column casing
    assert y.iloc[0] == 1  # Target mapping (funded -> 1)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model artifact not found")
def test_model_inference():
    """
    Integration test for the Scikit-Learn pipeline.
    Ensures the model can handle a new DataFrame and return a probability.
    """
    pipeline = joblib.load(MODEL_PATH)

    # Input matching the multi-column requirements of the pipeline
    test_input = pd.DataFrame(
        {
            "description_translated": [
                "This loan helps a small business owner buy inventory."
            ],
            "loan_amount": [750],
            "lender_term": [18],
            "sector_name": ["Retail"],
            "activity_name": ["General Store"],
        }
    )

    # Run inference
    proba = pipeline.predict_proba(test_input)

    # Assertions
    assert proba.shape == (1, 2)  # Returns binary classification probabilities
    assert 0 <= proba[0][1] <= 1  # Probability is between 0 and 1
