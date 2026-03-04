# AI Phishing Detection System

Classify text (e.g. URLs, emails) as **phishing** or **legitimate** using TF-IDF + Naive Bayes.

## Setup

```bash
pip install -r requirements.txt
```

## Train

Uses `dataset.csv` (columns: `text`, `label`). Saves a single pipeline to `model.pkl`.

```bash
python train_model.py
```

Output: accuracy, precision, recall, F1, classification report, confusion matrix, and 3-fold cross-validation score.

## Predict (CLI)

```bash
# Single string
python predict.py "Click here to claim your prize now!"

# From file
python predict.py --file sample.txt
```

## API (FastAPI)

Install dependencies (if not already done):

```bash
pip install -r requirements.txt
```

Start the API with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Or directly (for local dev):

```bash
python app.py
```

### API endpoints (versioned)

- **GET** `/api/v1/health` — basic health check  
- **GET** `/api/v1/readiness` — includes model load status and metadata  
- **POST** `/api/v1/predict` — classify a single text  
- **POST** `/api/v1/predict/batch` — classify a list of texts  

### Interactive docs

Once the server is running, open:

- Swagger UI: `http://localhost:8000/docs`  
- ReDoc: `http://localhost:8000/redoc`

### Example requests

Single prediction:

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Claim your free gift now\"}"
```

Batch prediction:

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"Claim your free gift now\", \"Meeting rescheduled to tomorrow\"]}"
```

## Project layout

| File           | Purpose                          |
|----------------|----------------------------------|
| `dataset.csv`  | Labeled text (text, label)        |
| `train_model.py` | Train and save pipeline         |
| `model.pkl`    | Saved pipeline (create by training) |
| `predict.py`   | CLI classifier                   |
| `app.py`       | Flask API                        |
| `requirements.txt` | Dependencies                 |
