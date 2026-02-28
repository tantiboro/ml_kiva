import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Modular imports
from ml_kiva.preprocess import load_and_clean_data, get_preprocessor
from ml_kiva.report import plot_learning_curve, print_feature_importance

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "ml-kiva-processed" / "processed_loans_balanced.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "kiva_pipeline.joblib"
# Add these lines to debug exactly what Python sees
print(f"DEBUG: BASE_DIR is {BASE_DIR}")
print(f"DEBUG: Looking for file at {DATA_PATH}")
print(f"DEBUG: File exists? {DATA_PATH.exists()}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ File not found at: {DATA_PATH.absolute()}")


def run_training():
    # --- 1. DETECT ENVIRONMENT ---
    # Check if we are on the Pixelbook ('penguin') or in a high-power Cloud environment
    is_local = os.uname().nodename == 'penguin'
    
    # --- 2. PREPARE DATA ---
    print(f"📂 Loading data from {DATA_PATH}...")
    X, y = load_and_clean_data(DATA_PATH)
    
    # Adaptive Downsampling
    if is_local and len(X) > 20000:
        print("🔬 Local dev detected: Limiting to 20k rows for stability.")
        X = X.sample(20000, random_state=42)
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # --- 3. ADAPTIVE HYPERPARAMETERS ---
    # Scale model complexity and parallelization based on the machine
    if is_local:
        n_est = 50
        depth = 10
        jobs = 1  # Save RAM on Pixelbook
    else:
        n_est = 200
        depth = None # Allow full growth on Colab
        jobs = -1 # Use all cores on Cloud

    print(f"⚙️ Config: trees={n_est}, depth={depth}, jobs={jobs}")

    # 4. Setup Pipeline
    preprocessor = get_preprocessor(
        "description_translated",
        ["loan_amount", "lender_term"],
        ["sector_name", "activity_name"],
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("rf", RandomForestClassifier(n_estimators=n_est, max_depth=depth, n_jobs=jobs, random_state=42))
    ])

    # 5. Train
    print("🏋️ Training model...")
    pipeline.fit(X_train, y_train)

    # # ... [Rest of your reporting code] ...
    # # 3. Train
    # print("🏋️ Training model...")
    # pipeline.fit(X_train, y_train)

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
