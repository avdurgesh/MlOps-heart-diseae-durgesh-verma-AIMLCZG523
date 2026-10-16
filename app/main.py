"""FastAPI service exposing the heart disease model."""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.predict import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))

REQUESTS = Counter("api_requests_total", "Total API requests", ["method", "path", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["path"])
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["label"])

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0", lifespan=lifespan)


class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, examples=[1])
    cp: int = Field(..., ge=1, le=4, examples=[1])
    trestbps: float = Field(..., ge=50, le=300, examples=[145])
    chol: float = Field(..., ge=50, le=700, examples=[233])
    fbs: int = Field(..., ge=0, le=1, examples=[1])
    restecg: int = Field(..., ge=0, le=2, examples=[2])
    thalach: float = Field(..., ge=40, le=250, examples=[150])
    exang: int = Field(..., ge=0, le=1, examples=[0])
    oldpeak: float = Field(..., ge=0, le=10, examples=[2.3])
    slope: int = Field(..., ge=1, le=3, examples=[3])
    ca: int = Field(..., ge=0, le=4, examples=[0])
    thal: int = Field(..., ge=3, le=7, examples=[6])


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probability_disease: float


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        logger.info("%s %s -> %s (%.1f ms)", request.method, path, status, elapsed * 1000)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(patient: PatientData):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = predict(state["model"], [patient.model_dump()])[0]
    PREDICTIONS.labels(str(result["prediction"])).inc()
    logger.info("prediction=%s confidence=%.3f", result["prediction"], result["confidence"])
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
