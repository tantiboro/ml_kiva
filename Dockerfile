# 1. Use a slim Python image to keep the footprint small
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies for spaCy and Scikit-Learn
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Download the specific spaCy model used in your preprocessing
RUN python -m spacy download en_core_web_sm

# 6. Copy the modular project structure
# We only copy what's needed for the app to run
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .
COPY pyproject.toml .

# 7. Install the project in editable mode to link the src package
RUN pip install -e .

# 8. Expose the Streamlit port (Default 8501)
EXPOSE 8080

# 9. Configure Streamlit for Cloud Run (Headless and Port mapping)
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
