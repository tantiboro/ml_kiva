import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import your new modular functions
from ml_kiva.preprocess import load_and_clean_data, get_preprocessor
from ml_kiva.report import plot_learning_curve, print_feature_importance

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "processed_loans_balanced.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "kiva_pipeline.joblib"
# Add these lines to debug exactly what Python sees
print(f"DEBUG: BASE_DIR is {BASE_DIR}")
print(f"DEBUG: Looking for file at {DATA_PATH}")
print(f"DEBUG: File exists? {DATA_PATH.exists()}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ File not found at: {DATA_PATH.absolute()}")


def run_training():
    # 1. Prepare Data
    X, y = load_and_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # 2. Setup Pipeline
    preprocessor = get_preprocessor(
        "description_translated",
        ["loan_amount", "lender_term"],
        ["sector_name", "activity_name"],
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "rf",
                RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
            ),
        ]
    )

    # 3. Train
    print("🏋️ Training model...")
    pipeline.fit(X_train, y_train)

    # 4. Report & Diagnostics
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plot_learning_curve(pipeline, X_train, y_train, MODEL_DIR / "learning_curve.png")
    print_feature_importance(pipeline)

    print("\n📊 Final Performance Report:")
    print(classification_report(y_test, pipeline.predict(X_test)))

    # 5. Save
    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    run_training()
