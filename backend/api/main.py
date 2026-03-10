"""
Crop Recommendation API
=======================
FastAPI backend for crop recommendation system.

Endpoints:
- POST /predict         — Single crop prediction
- POST /predict-batch   — CSV file batch prediction
- GET  /history         — Prediction history from MongoDB
- POST /retrain         — Trigger model retraining
- GET  /health          — Health check
- GET  /model-info      — Loaded model metadata
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import joblib
import numpy as np
import pandas as pd
import os
import io
import json
import subprocess
import sys
from datetime import datetime
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class PredictionInput(BaseModel):
    nitrogen: float = Field(..., ge=0, le=200, description="Nitrogen content (0-200)")
    phosphorus: float = Field(..., ge=0, le=200, description="Phosphorus content (0-200)")
    potassium: float = Field(..., ge=0, le=200, description="Potassium content (0-200)")

    class Config:
        json_schema_extra = {
            "example": {"nitrogen": 90, "phosphorus": 42, "potassium": 43}
        }


class PredictionOutput(BaseModel):
    success: bool
    input: Dict[str, float]
    prediction: Dict[str, Any]
    timestamp: str


class BatchPredictionOutput(BaseModel):
    success: bool
    predictions: List[Dict[str, Any]]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: str


class ModelInfo(BaseModel):
    model_name: str
    features: List[str]
    classes: List[str]
    accuracy: float
    trained_date: str
    model_comparison: Optional[Dict[str, Any]] = None


# ============================================================================
# App Setup
# ============================================================================

app = FastAPI(
    title="Crop Recommendation API",
    description="ML-powered API for recommending crops based on soil nutrients",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Globals
# ============================================================================

model = None
scaler = None
label_encoder = None
model_metadata = None

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "crop_recommendation_db"
COLLECTION_NAME = "predictions"

mongodb_client: Optional[AsyncIOMotorClient] = None
database = None
collection = None

# Models that need scaled features
SCALED_MODELS = {'KNN', 'Logistic Regression', 'SVM', 'MLP Neural Network', 'XGBoost', 'Gradient Boosting'}


# ============================================================================
# Helpers
# ============================================================================

def load_model_artifacts():
    """Load model, scaler, label_encoder, and metadata from disk."""
    global model, scaler, label_encoder, model_metadata

    model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_model')

    model = joblib.load(os.path.join(model_dir, 'crop_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

    le_path = os.path.join(model_dir, 'label_encoder.pkl')
    if os.path.exists(le_path):
        label_encoder = joblib.load(le_path)

    with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
        model_metadata = json.load(f)

    logger.info(f"✓ Model loaded: {model_metadata['model_name']} "
                f"(accuracy={model_metadata['accuracy']:.4f})")


def predict_single(n: float, p: float, k: float):
    """Run prediction for a single sample. Returns (crop, confidence)."""
    features = np.array([[n, p, k]])
    model_name = model_metadata.get('model_name', '')
    uses_scaling = model_metadata.get('uses_scaling', model_name in SCALED_MODELS)

    if uses_scaling and scaler is not None:
        features = scaler.transform(features)

    if model_name == 'XGBoost' and label_encoder is not None:
        pred_encoded = model.predict(features)[0]
        crop = label_encoder.inverse_transform([pred_encoded])[0]
    else:
        crop = str(model.predict(features)[0])

    # Confidence
    confidence = 0.85
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(features)
            confidence = float(np.max(probabilities))
        except Exception:
            pass

    return crop, round(confidence, 4)


async def save_to_database(input_data: dict, prediction: str, confidence: float):
    """Save prediction to MongoDB."""
    if collection is None:
        return None
    try:
        document = {
            "nitrogen": input_data["nitrogen"],
            "phosphorus": input_data["phosphorus"],
            "potassium": input_data["potassium"],
            "recommended_crop": prediction,
            "confidence": confidence,
            "timestamp": datetime.utcnow(),
        }
        result = await collection.insert_one(document)
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        return None


# ============================================================================
# Lifecycle
# ============================================================================

@app.on_event("startup")
async def startup_event():
    global mongodb_client, database, collection

    logger.info("Starting Crop Recommendation API v2...")

    # Load ML artifacts
    try:
        load_model_artifacts()
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        raise

    # Connect MongoDB
    try:
        mongodb_client = AsyncIOMotorClient(MONGODB_URL)
        database = mongodb_client[DATABASE_NAME]
        collection = database[COLLECTION_NAME]
        await mongodb_client.admin.command('ping')
        logger.info(f"✓ Connected to MongoDB at {MONGODB_URL}")
    except Exception as e:
        logger.warning(f"⚠ MongoDB connection failed: {e}")
        logger.warning("API will continue without database functionality")


@app.on_event("shutdown")
async def shutdown_event():
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        logger.info("MongoDB connection closed")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Crop Recommendation API",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    db_connected = False
    if mongodb_client:
        try:
            await mongodb_client.admin.command('ping')
            db_connected = True
        except Exception:
            pass
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        database_connected=db_connected,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    return ModelInfo(
        model_name=model_metadata.get('model_name', 'Unknown'),
        features=model_metadata.get('feature_names', []),
        classes=model_metadata.get('classes', []),
        accuracy=model_metadata.get('accuracy', 0.0),
        trained_date=model_metadata.get('trained_date', 'Unknown'),
        model_comparison=model_metadata.get('model_comparison'),
    )


# ──────────────────────────── POST /predict ────────────────────────────

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_crop(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        crop, confidence = predict_single(
            input_data.nitrogen, input_data.phosphorus, input_data.potassium
        )

        input_dict = {
            "nitrogen": input_data.nitrogen,
            "phosphorus": input_data.phosphorus,
            "potassium": input_data.potassium,
        }

        await save_to_database(input_dict, crop, confidence)

        return PredictionOutput(
            success=True,
            input=input_dict,
            prediction={"recommended_crop": crop, "confidence_score": confidence},
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ──────────────────────────── POST /predict-batch ──────────────────────

@app.post("/predict-batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """Accept a CSV file and predict crops for each row."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {
        'nitrogen': 'nitrogen', 'n': 'nitrogen',
        'phosphorus': 'phosphorus', 'p': 'phosphorus',
        'potassium': 'potassium', 'k': 'potassium',
    }
    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

    required = ['nitrogen', 'phosphorus', 'potassium']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing}. CSV must contain nitrogen/N, phosphorus/P, potassium/K."
        )

    predictions = []
    for _, row in df.iterrows():
        n, p, k = float(row['nitrogen']), float(row['phosphorus']), float(row['potassium'])
        crop, confidence = predict_single(n, p, k)
        pred = {
            "nitrogen": n, "phosphorus": p, "potassium": k,
            "crop": crop, "confidence": confidence,
        }
        predictions.append(pred)

        # Save each to DB
        await save_to_database(
            {"nitrogen": n, "phosphorus": p, "potassium": k}, crop, confidence
        )

    return BatchPredictionOutput(
        success=True, predictions=predictions, count=len(predictions)
    )


# ──────────────────────────── GET /history ─────────────────────────────

@app.get("/history", tags=["History"])
async def get_prediction_history(limit: int = 50):
    if collection is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    try:
        limit = min(max(limit, 1), 500)
        cursor = collection.find().sort("timestamp", -1).limit(limit)
        history = []
        async for doc in cursor:
            doc['_id'] = str(doc['_id'])
            if 'timestamp' in doc and hasattr(doc['timestamp'], 'isoformat'):
                doc['timestamp'] = doc['timestamp'].isoformat()
            history.append(doc)

        return {"success": True, "count": len(history), "data": history}
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {e}")


# ──────────────────────────── POST /retrain ────────────────────────────

@app.post("/retrain", tags=["Model"])
async def retrain_model():
    """Trigger model retraining using stored prediction data."""
    retrain_script = os.path.join(
        os.path.dirname(__file__), '..', 'model', 'retrain_model.py'
    )

    if not os.path.exists(retrain_script):
        raise HTTPException(status_code=404, detail="Retrain script not found.")

    try:
        result = subprocess.run(
            [sys.executable, retrain_script],
            capture_output=True, text=True, timeout=300
        )

        success = result.returncode == 0

        if success:
            # Reload the updated model
            try:
                load_model_artifacts()
                logger.info("✓ Model reloaded after retraining")
            except Exception as e:
                logger.error(f"Model reload failed: {e}")

        return {
            "success": success,
            "message": "Model retrained successfully" if success else "Retraining failed",
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Retraining timed out.")
    except Exception as e:
        logger.error(f"Retrain error: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")


# ──────────────────────────── DELETE /history ──────────────────────────

@app.delete("/history", tags=["History"])
async def clear_history():
    if collection is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    try:
        result = await collection.delete_many({})
        return {
            "success": True,
            "message": f"Deleted {result.deleted_count} records",
            "deleted_count": result.deleted_count,
        }
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {e}")


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
