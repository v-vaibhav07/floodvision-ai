
from pathlib import Path
from io import BytesIO
import os
import sys
import csv
import time
import logging

import cv2
import joblib
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from frozen_kmeans import FrozenKMeans
from features import (
    extract_teacher_features,
    extract_student_features
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("floodvision")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model_bundle.joblib"
METRICS_PATH = BASE_DIR / "final_test_results_all_models.csv"

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp"
}


# --------------------------------------------------
# 2. INITIALIZE FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="FloodVision AI",
    description="Machine Learning Based Flood Area Detection",
    version="2.0.0"
)


# --------------------------------------------------
# 3. CORS CONFIGURATION - VERCEL + LOCALHOST
# --------------------------------------------------

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://floodvision-ai-rho.vercel.app"
)

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://floodvision-ai-rho.vercel.app"
]

# Supports one or more frontend URLs separated by commas.
for url in FRONTEND_URL.split(","):
    url = url.strip().rstrip("/")

    if url and url not in allowed_origins:
        allowed_origins.append(url)

logger.info("Allowed CORS origins: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "X-Flood-Percentage",
        "X-Inference-Time-Ms"
    ]
)


# --------------------------------------------------
# 4. LOAD TRAINED ML MODELS
# --------------------------------------------------

# Compatibility with FrozenKMeans exported from Colab.
# Only load trusted joblib model files.
sys.modules["__main__"].FrozenKMeans = FrozenKMeans

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model bundle not found: {MODEL_PATH}"
    )

logger.info("Loading FloodVision AI models...")

bundle = joblib.load(MODEL_PATH)

models = bundle["models"]
scalers = bundle["scalers"]

image_size = bundle.get("image_size", 256)

if isinstance(image_size, (tuple, list)):
    IMAGE_HEIGHT = int(image_size[0])
    IMAGE_WIDTH = int(image_size[1])
else:
    IMAGE_HEIGHT = int(image_size)
    IMAGE_WIDTH = int(image_size)

if IMAGE_HEIGHT <= 0 or IMAGE_WIDTH <= 0:
    raise ValueError("Invalid model image dimensions")

logger.info("Models loaded successfully: %s", len(models))


# --------------------------------------------------
# 5. BATCH PREDICTION
# --------------------------------------------------

def predict_batched(model, features, batch_size=2048):

    predictions = []

    for start in range(0, len(features), batch_size):

        batch = features[start:start + batch_size]

        probabilities = model.predict_proba(batch)

        predictions.append(probabilities[:, 1])

    return np.concatenate(predictions)


# --------------------------------------------------
# 6. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "FloodVision AI API is running",
        "status": "success",
        "version": "2.0.0",
        "total_models": len(models),
        "endpoints": [
            "/health",
            "/models",
            "/metrics",
            "/predict",
            "/docs"
        ]
    }


# --------------------------------------------------
# 7. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "models_loaded": len(models),
        "metrics_available": METRICS_PATH.exists()
    }


# --------------------------------------------------
# 8. GET AVAILABLE MODELS
# --------------------------------------------------

@app.get("/models")
def get_models():

    return {
        "total_models": len(models),
        "models": list(models.keys())
    }


# --------------------------------------------------
# 9. GET MODEL EVALUATION METRICS
# --------------------------------------------------

@app.get("/metrics")
def get_metrics():

    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results CSV not found"
        )

    results = []

    try:
        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                results.append({
                    "model": row["Model"],
                    "family": row["Family"],
                    "precision": float(
                        row["Precision (%)"]
                    ),
                    "recall": float(
                        row["Recall (%)"]
                    ),
                    "f1": float(
                        row["F1 (%)"]
                    ),
                    "iou": float(
                        row["IoU (%)"]
                    ),
                    "size_kb": float(
                        row["Size (KB)"]
                    ),
                    "time_ms": float(
                        row["Time (ms/image)"]
                    )
                })

    except (KeyError, ValueError, TypeError, csv.Error) as error:

        logger.exception("Failed to read metrics")

        raise HTTPException(
            status_code=500,
            detail="Invalid evaluation results CSV"
        ) from error

    results.sort(
        key=lambda item: item["iou"],
        reverse=True
    )

    best_model = (
        results[0]["model"]
        if results else None
    )

    fastest_model = (
        min(
            results,
            key=lambda item: item["time_ms"]
        )["model"]
        if results else None
    )

    return {
        "total_models": len(results),
        "best_model": best_model,
        "fastest_model": fastest_model,
        "results": results
    }


# --------------------------------------------------
# 10. FLOOD PREDICTION API
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = "RF-T"
):

    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}"
        )

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Upload a PNG, JPEG or WebP image"
        )

    try:
        data = await file.read(MAX_FILE_SIZE + 1)
    finally:
        await file.close()

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB"
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Empty image file"
        )

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    spec = models[model_name]
    feature_type = spec["feature_type"]

    start_time = time.perf_counter()

    try:

        # ------------------------------------------
        # FEATURE EXTRACTION
        # ------------------------------------------

        if feature_type in ("teacher", "km_teacher"):

            features = extract_teacher_features(image)

        else:

            features = extract_student_features(image)

        # ------------------------------------------
        # FEATURE SCALING
        # ------------------------------------------

        features = scalers[feature_type].transform(
            features
        ).astype(np.float32)

        # ------------------------------------------
        # SUPERVISED PREDICTION
        # ------------------------------------------

        if spec["kind"] == "supervised":

            probabilities = predict_batched(
                spec["model"],
                features
            )

            threshold = spec["threshold"]

            prediction = (
                probabilities >= threshold
            ).astype(np.uint8)

        # ------------------------------------------
        # K-MEANS PREDICTION
        # ------------------------------------------

        else:

            clusters = spec["model"].predict(
                features
            )

            mapping = np.asarray(
                spec["mapping"]
            )

            prediction = mapping[clusters].astype(
                np.uint8
            )

        # ------------------------------------------
        # CREATE SEGMENTATION MASK
        # ------------------------------------------

        mask = prediction.reshape(
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        )

        inference_time = (
            time.perf_counter() - start_time
        ) * 1000

        flood_percentage = float(
            np.mean(mask) * 100
        )

        # ------------------------------------------
        # GENERATE FLOOD OVERLAY
        # ------------------------------------------

        overlay = image.copy()

        flood_color = np.array(
            [255, 45, 149],
            dtype=np.float32
        )

        overlay[mask == 1] = (
            0.5 * image[mask == 1] +
            0.5 * flood_color
        ).astype(np.uint8)

        # ------------------------------------------
        # ORIGINAL + SEGMENTATION RESULT
        # ------------------------------------------

        result = np.concatenate(
            [image, overlay],
            axis=1
        )

        result_bgr = cv2.cvtColor(
            result,
            cv2.COLOR_RGB2BGR
        )

        success, encoded = cv2.imencode(
            ".png",
            result_bgr
        )

        if not success:
            raise RuntimeError(
                "Image encoding failed"
            )

    except Exception as error:

        logger.exception(
            "Prediction failed for model %s",
            model_name
        )

        raise HTTPException(
            status_code=500,
            detail="Flood prediction failed"
        ) from error

    return StreamingResponse(
        BytesIO(encoded.tobytes()),
        media_type="image/png",
        headers={
            "X-Flood-Percentage": str(
                round(flood_percentage, 2)
            ),
            "X-Inference-Time-Ms": str(
                round(inference_time, 2)
            )
        }
    )
