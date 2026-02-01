Kiva Loan Funding Predictor
Author: Tantiboro "Tanti" Ouattara

Stack: Python, spaCy NLP, Scikit-Learn, Streamlit, Docker, GCP Cloud Run

📌 Project Overview
This project leverages Natural Language Processing (NLP) to predict the success of microlending requests on the Kiva platform. By moving beyond simple keyword matching to linguistic lemmatization, the model understands the core intent of loan descriptions to identify if a request will be "Fully Funded" or "Expired."

This repository represents a transition from an experimental notebook to a production-grade ML pipeline with automated CI/CD and high-performance text processing.

🏗️ Architecture & Pipeline
NLP Engine: Advanced text cleaning via spaCy, including HTML stripping, non-ASCII removal, and multi-core lemmatization.

Modeling: A Scikit-Learn Pipeline integrating TfidfVectorizer and a RandomForestClassifier for seamless inference.

Performance: Implements nlp.pipe for 5x–10x faster batch processing during training and reporting.

Frontend: A Streamlit dashboard featuring real-time health checks and "feature importance" explainability.

DevOps: * CI: GitHub Actions for unit testing (pytest) and automated EDA report generation.

CD: Automated containerization (Docker) and deployment to GCP Cloud Run (2Gi RAM optimized).

📂 Project Structure
Plaintext

├── .github/workflows/
│   └── main.yml           # CI/CD pipeline (GCP Deployment & Artifacts)
├── data/                  # Local data storage (Git ignored)
├── models/                # Saved Pipeline (.joblib) artifacts
├── src/
│   ├── preprocess.py      # High-performance spaCy cleaning logic
│   ├── train.py           # Parallelized training & serialization
│   └── report.py          # Automated NLP & Numerical EDA
├── app.py                 # Streamlit UI with Model Health Checks
├── Dockerfile             # Production container (Python 3.10-slim)
├── .dockerignore          # Optimization to keep image size small
└── requirements.txt       # Streamlined dependency list
🚀 Getting Started
1. Local Environment Setup
Since this project uses specialized NLP tools, it is recommended to use a clean virtual environment:

Bash

# Create and activate environment
python -m venv nlp_venv
source nlp_venv/bin/activate  # Linux/WSL
# .\.venv\Scripts\activate   # Windows

# Install requirements
pip install -r requirements.txt

# Download the required NLP model
python -m spacy download en_core_web_sm

# Run the dashboard
streamlit run app.py
2. High-Performance Training
To rebuild the model using all available CPU cores:

Bash

python src/train.py
📊 CI/CD & Cloud Deployment
This project is configured for Continuous Deployment to Google Cloud Run.

Tests: Every push triggers pytest to verify the NLP cleaning logic.

Reports: The pipeline generates updated Word Clouds and Correlation Heatmaps as downloadable GitHub artifacts.

Deploy: The app is containerized and pushed to GCR.

Note: Cloud Run is configured with 2Gi memory to support spaCy's linguistic models.

📈 Key Dashboard Features
Health Check: Automatically verifies the existence of ML artifacts and spaCy models on startup.

Linguistic Explainability: When a prediction is made, the app visualizes which specific words (lemmas) most influenced the model's decision.

Automated EDA: Real-time visualization of loan status distributions and sector trends.

Final Next Step
Would you like me to show you how to add a "Model Versioning" tag to your train.py so that your app.py can display the exact date the model was last trained?