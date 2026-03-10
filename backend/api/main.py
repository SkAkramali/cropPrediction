"""
Crop Recommendation API
=======================
FastAPI backend for crop recommendation system.

Endpoints:
- POST /predict: Get crop recommendation based on soil nutrients
- GET /health: Health check endpoint
- GET /model-info: Get information about the loaded model
- GET /history: Get prediction history from database

Author: AI Assistant
Date: March 2026
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import joblib
import numpy as np
import os
from datetime import datetime
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class PredictionInput(BaseModel):
    """Input model for crop prediction"""
    nitrogen: float = Field(..., ge=0, le=200, description="Nitrogen content in soil (0-200)")
    phosphorus: float = Field(..., ge=0, le=200, description="Phosphorus content in soil (0-200)")
    potassium: float = Field(..., ge=0, le=200, description="Potassium content in soil (0-200)")
    
    @validator('nitrogen', 'phosphorus', 'potassium')
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError('Value must be positive')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "nitrogen": 90,
                "phosphorus": 42,
                "potassium": 43
            }
        }


class PredictionOutput(BaseModel):
    """Output model for crop prediction"""
    success: bool
    input: Dict[str, float]
    prediction: Dict[str, Any]
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "input": {
                    "nitrogen": 90,
                    "phosphorus": 42,
                    "potassium": 43
                },
                "prediction": {
                    "recommended_crop": "rice",
                    "confidence_score": 0.92
                },
                "timestamp": "2026-03-10T12:00:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: str


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    features: List[str]
    classes: List[str]
    accuracy: float
    trained_date: str


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Crop Recommendation API",
    description="ML-powered API for recommending crops based on soil nutrients",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# CORS middleware - allows frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global Variables and Configuration
# ============================================================================

# Model and scaler
model = None
scaler = None
model_metadata = None

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "crop_recommendation_db"
COLLECTION_NAME = "predictions"

# MongoDB client
mongodb_client: Optional[AsyncIOMotorClient] = None
database = None
collection = None


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model and database connection on startup"""
    global model, scaler, model_metadata, mongodb_client, database, collection
    
    logger.info("Starting Crop Recommendation API...")
    
    # Load ML model
    try:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_model')
        
        model_path = os.path.join(model_dir, 'crop_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        
        # Load model
        model = joblib.load(model_path)
        logger.info(f"✓ Model loaded successfully from {model_path}")
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        logger.info(f"✓ Scaler loaded successfully from {scaler_path}")
        
        # Load metadata
        import json
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
        logger.info(f"✓ Model metadata loaded: {model_metadata['model_name']}")
        
    except Exception as e:
        logger.error(f"✗ Error loading model: {str(e)}")
        raise
    
    # Connect to MongoDB
    try:
        mongodb_client = AsyncIOMotorClient(MONGODB_URL)
        database = mongodb_client[DATABASE_NAME]
        collection = database[COLLECTION_NAME]
        
        # Test connection
        await mongodb_client.admin.command('ping')
        logger.info(f"✓ Connected to MongoDB at {MONGODB_URL}")
        
    except Exception as e:
        logger.warning(f"⚠ MongoDB connection failed: {str(e)}")
        logger.warning("API will continue without database functionality")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global mongodb_client
    
    if mongodb_client:
        mongodb_client.close()
        logger.info("MongoDB connection closed")
    
    logger.info("Crop Recommendation API shutdown complete")


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_confidence(model, features, prediction):
    """
    Calculate confidence score for the prediction
    
    Args:
        model: Trained model
        features: Input features
        prediction: Model prediction
    
    Returns:
        float: Confidence score (0-1)
    """
    try:
        # For models with predict_proba method
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            confidence = float(np.max(probabilities))
        else:
            # For models without probability (e.g., some Decision Trees)
            confidence = 0.85  # Default confidence
        
        return round(confidence, 4)
    
    except Exception as e:
        logger.warning(f"Could not calculate confidence: {str(e)}")
        return 0.80


async def save_to_database(input_data: dict, prediction: str, confidence: float):
    """
    Save prediction to MongoDB
    
    Args:
        input_data: Input soil nutrient values
        prediction: Predicted crop name
        confidence: Confidence score
    """
    if collection is None:
        logger.warning("Database not connected. Skipping save.")
        return None
    
    try:
        document = {
            "nitrogen": input_data["nitrogen"],
            "phosphorus": input_data["phosphorus"],
            "potassium": input_data["potassium"],
            "recommended_crop": prediction,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        }
        
        result = await collection.insert_one(document)
        logger.info(f"Prediction saved to database with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
        return None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Crop Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse: System health status
    """
    db_connected = False
    
    if mongodb_client:
        try:
            await mongodb_client.admin.command('ping')
            db_connected = True
        except:
            pass
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        database_connected=db_connected,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model
    
    Returns:
        ModelInfo: Model metadata and information
    """
    if model_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata not available"
        )
    
    return ModelInfo(
        model_name=model_metadata.get('model_name', 'Unknown'),
        features=model_metadata.get('feature_names', []),
        classes=model_metadata.get('classes', []),
        accuracy=model_metadata.get('accuracy', 0.0),
        trained_date=model_metadata.get('trained_date', 'Unknown')
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_crop(input_data: PredictionInput):
    """
    Predict the best crop to grow based on soil nutrient values
    
    Args:
        input_data: Soil nutrient values (N, P, K)
    
    Returns:
        PredictionOutput: Crop recommendation with confidence score
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator."
        )
    
    try:
        # Prepare input features
        features = np.array([[
            input_data.nitrogen,
            input_data.phosphorus,
            input_data.potassium
        ]])
        
        # Determine if scaling is needed based on model type
        model_name = model_metadata.get('model_name', '')
        if model_name in ['KNN', 'Logistic Regression'] and scaler is not None:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            confidence = calculate_confidence(model, features_scaled, prediction)
        else:
            prediction = model.predict(features)[0]
            confidence = calculate_confidence(model, features, prediction)
        
        # Convert prediction to string (in case it's not)
        recommended_crop = str(prediction)
        
        # Prepare response
        input_dict = {
            "nitrogen": input_data.nitrogen,
            "phosphorus": input_data.phosphorus,
            "potassium": input_data.potassium
        }
        
        # Save to database (async, non-blocking)
        await save_to_database(input_dict, recommended_crop, confidence)
        
        # Return prediction
        response = PredictionOutput(
            success=True,
            input=input_dict,
            prediction={
                "recommended_crop": recommended_crop,
                "confidence_score": confidence
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Prediction made: {recommended_crop} (confidence: {confidence})")
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/history", tags=["History"])
async def get_prediction_history(limit: int = 10):
    """
    Get recent prediction history from database
    
    Args:
        limit: Number of records to return (default: 10, max: 100)
    
    Returns:
        List of recent predictions
    """
    if collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not connected"
        )
    
    try:
        # Limit check
        limit = min(limit, 100)
        
        # Query database
        cursor = collection.find().sort("timestamp", -1).limit(limit)
        history = []
        
        async for document in cursor:
            # Convert ObjectId to string
            document['_id'] = str(document['_id'])
            # Convert datetime to ISO format
            document['timestamp'] = document['timestamp'].isoformat()
            history.append(document)
        
        return {
            "success": True,
            "count": len(history),
            "data": history
        }
    
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch history: {str(e)}"
        )


@app.delete("/history", tags=["History"])
async def clear_history():
    """
    Clear all prediction history from database
    
    Returns:
        Confirmation message
    """
    if collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not connected"
        )
    
    try:
        result = await collection.delete_many({})
        
        return {
            "success": True,
            "message": f"Deleted {result.deleted_count} records",
            "deleted_count": result.deleted_count
        }
    
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear history: {str(e)}"
        )


# ============================================================================
# Run the API
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
