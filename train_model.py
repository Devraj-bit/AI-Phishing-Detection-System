import pandas as pd
import pickle
from pathlib import Path
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Load dataset
def _is_zip_file(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    with p.open("rb") as f:
        return f.read(2) == b"PK"


def read_dataset_csv(path: str) -> pd.DataFrame:
    """
    Loads a dataset CSV that may be either:
    - a normal CSV file, OR
    - a ZIP archive (misnamed as .csv) containing one or more CSV files.
    """
    last_err: Exception | None = None
    encodings = ("utf-8", "cp1252", "latin-1")

    if _is_zip_file(path):
        with ZipFile(path) as zf:
            inner = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None) or zf.namelist()[0]
            with zf.open(inner) as fp:
                for enc in encodings:
                    try:
                        return pd.read_csv(
                            fp,
                            encoding=enc,
                            encoding_errors="replace",
                            engine="python",
                            on_bad_lines="skip",
                        )
                    except Exception as e:
                        last_err = e
                        fp.seek(0)
        raise last_err  # type: ignore[misc]

    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                encoding_errors="replace",
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as e:
            last_err = e
    raise last_err  # type: ignore[misc]


data = read_dataset_csv("dataset.csv")
def _norm_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


col_map = {_norm_col(c): c for c in data.columns}

text_col = (
    col_map.get("text")
    or col_map.get("emailtext")
    or col_map.get("emailbody")
    or col_map.get("body")
    or col_map.get("message")
    or col_map.get("content")
)
label_col = (
    col_map.get("label")
    or col_map.get("emailtype")
    or col_map.get("class")
    or col_map.get("category")
    or col_map.get("type")
)

if text_col is None or label_col is None:
    raise KeyError(
        f"Could not find required columns. Found columns: {list(data.columns)}. "
        f"Need a text column (e.g. 'text'/'Email Text') and label column (e.g. 'label'/'Email Type')."
    )

data = data.dropna(subset=[text_col, label_col])
X = data[text_col].astype(str)
y = data[label_col].astype(str)

# Split first so we don't leak test data into vectorizer
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: vectorize then classify
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB()),
])
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Metrics
print("=== Test Set Metrics ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, average="weighted"), 4))
print("Recall:", round(recall_score(y_test, y_pred, average="weighted"), 4))
print("F1 Score:", round(f1_score(y_test, y_pred, average="weighted"), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation on training set
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3)
print(f"\nCross-Validation (3-fold) Accuracy: {cv_scores.mean():.4} (+/- {cv_scores.std() * 2:.4})")

# Save pipeline (includes vectorizer + model)
out_dir = Path(__file__).resolve().parent
with open(out_dir / "model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("\nModel saved to model.pkl (pipeline: vectorizer + classifier)")
