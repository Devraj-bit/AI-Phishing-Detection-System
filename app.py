"""
Advanced FastAPI-based API for phishing detection.

Run (development):
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Docs:
  - Swagger UI:       /docs
  - ReDoc:            /redoc
  - OpenAPI schema:   /openapi.json
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


APP_VERSION = "1.0.0"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


class PredictionRequest(BaseModel):
    text: str = Field(..., description="Text or URL to classify", min_length=1)


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts or URLs to classify", min_items=1)


class PredictionResponse(BaseModel):
    prediction: str
    text_length: int
    model_version: Optional[str] = Field(
        default=None, description="Version of the model or API serving the prediction"
    )


class BatchPredictionResponseItem(BaseModel):
    prediction: str
    text_length: int


class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionResponseItem]
    count: int
    model_version: Optional[str] = None


class ModelInfo(BaseModel):
    model_loaded: bool
    model_path: str
    app_version: str


app = FastAPI(
    title="AI Phishing Detection API",
    version=APP_VERSION,
    description=(
        "Advanced API for classifying text as phishing or legitimate using a "
        "TF-IDF + Naive Bayes model."
    ),
)

# Allow CORS for local development; tighten origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_pipeline = None


def _load_model():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl not found. Run train_model.py first.")

    with MODEL_PATH.open("rb") as f:
        _pipeline = pickle.load(f)
    return _pipeline


@app.on_event("startup")
def _startup_event():
    """
    Attempt to load the model at startup so readiness checks can detect issues early.
    """
    try:
        _load_model()
    except FileNotFoundError:
        # Model is optional at startup; readiness endpoint will surface the issue.
        pass


@app.get("/api/v1/health", tags=["system"])
def health() -> dict:
    """
    Basic liveness check.
    """
    return {"status": "ok"}


@app.get("/api/v1/readiness", response_model=ModelInfo, tags=["system"])
def readiness() -> ModelInfo:
    """
    Readiness check that verifies if the model is loaded and available.
    """
    try:
        model_loaded = _load_model() is not None
    except FileNotFoundError:
        model_loaded = False

    return ModelInfo(
        model_loaded=model_loaded,
        model_path=str(MODEL_PATH),
        app_version=APP_VERSION,
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    status_code=status.HTTP_200_OK,
)
def predict(req: PredictionRequest) -> PredictionResponse:
    """
    Predict whether a single text is phishing or legitimate.
    """
    try:
        pipeline = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    label = pipeline.predict([req.text])[0]
    return PredictionResponse(
        prediction=str(label),
        text_length=len(req.text),
        model_version=APP_VERSION,
    )


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
    status_code=status.HTTP_200_OK,
)
def predict_batch(req: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict labels for a batch of texts.
    """
    try:
        pipeline = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    preds = pipeline.predict(req.texts)
    items = [
        BatchPredictionResponseItem(prediction=str(label), text_length=len(text))
        for label, text in zip(preds, req.texts)
    ]
    return BatchPredictionResponse(
        predictions=items,
        count=len(items),
        model_version=APP_VERSION,
    )


if __name__ == "__main__":
    # Convenient entrypoint for local development.
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
