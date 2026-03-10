"""
MongoDB Configuration Module
=============================
Configuration and utility functions for MongoDB connection.

This module provides:
- Database connection setup
- Collection management
- Query utilities
- Data validation

Author: AI Assistant
Date: March 2026
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os
from typing import Optional, Dict, List
from datetime import datetime
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class MongoDBConfig:
    """MongoDB configuration class"""
    
    def __init__(
        self,
        mongodb_url: Optional[str] = None,
        database_name: str = "crop_recommendation_db",
        collection_name: str = "predictions"
    ):
        """
        Initialize MongoDB configuration
        
        Args:
            mongodb_url: MongoDB connection URL
            database_name: Name of the database
            collection_name: Name of the collection
        """
        self.mongodb_url = mongodb_url or os.getenv(
            "MONGODB_URL",
            "mongodb://localhost:27017"
        )
        self.database_name = database_name
        self.collection_name = collection_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.collection = None
    
    async def connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.mongodb_url)
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"✓ Connected to MongoDB at {self.mongodb_url}")
            logger.info(f"✓ Using database: {self.database_name}")
            logger.info(f"✓ Using collection: {self.collection_name}")
            
            return True
        
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"✗ Failed to connect to MongoDB: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"✗ Unexpected error connecting to MongoDB: {str(e)}")
            return False
    
    async def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def get_collection(self):
        """Get the collection object"""
        return self.collection


# ============================================================================
# Database Operations
# ============================================================================

class PredictionDatabase:
    """Database operations for crop predictions"""
    
    def __init__(self, config: MongoDBConfig):
        """
        Initialize database operations
        
        Args:
            config: MongoDBConfig instance
        """
        self.config = config
        self.collection = config.collection
    
    async def insert_prediction(
        self,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        recommended_crop: str,
        confidence: float
    ) -> Optional[str]:
        """
        Insert a prediction record into the database
        
        Args:
            nitrogen: Nitrogen value
            phosphorus: Phosphorus value
            potassium: Potassium value
            recommended_crop: Predicted crop name
            confidence: Confidence score
        
        Returns:
            Inserted document ID or None if failed
        """
        try:
            document = {
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "recommended_crop": recommended_crop,
                "confidence": confidence,
                "timestamp": datetime.utcnow()
            }
            
            result = await self.collection.insert_one(document)
            logger.info(f"Prediction inserted with ID: {result.inserted_id}")
            
            return str(result.inserted_id)
        
        except Exception as e:
            logger.error(f"Error inserting prediction: {str(e)}")
            return None
    
    async def get_all_predictions(self, limit: int = 100) -> List[Dict]:
        """
        Get all predictions from the database
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of prediction documents
        """
        try:
            cursor = self.collection.find().sort("timestamp", -1).limit(limit)
            predictions = []
            
            async for document in cursor:
                document['_id'] = str(document['_id'])
                document['timestamp'] = document['timestamp'].isoformat()
                predictions.append(document)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error fetching predictions: {str(e)}")
            return []
    
    async def get_predictions_by_crop(self, crop_name: str) -> List[Dict]:
        """
        Get predictions for a specific crop
        
        Args:
            crop_name: Name of the crop
        
        Returns:
            List of predictions for the specified crop
        """
        try:
            cursor = self.collection.find(
                {"recommended_crop": crop_name}
            ).sort("timestamp", -1)
            
            predictions = []
            
            async for document in cursor:
                document['_id'] = str(document['_id'])
                document['timestamp'] = document['timestamp'].isoformat()
                predictions.append(document)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error fetching predictions by crop: {str(e)}")
            return []
    
    async def get_statistics(self) -> Dict:
        """
        Get statistics about predictions
        
        Returns:
            Dictionary containing statistics
        """
        try:
            # Total predictions
            total_count = await self.collection.count_documents({})
            
            # Predictions by crop
            pipeline = [
                {
                    "$group": {
                        "_id": "$recommended_crop",
                        "count": {"$sum": 1},
                        "avg_confidence": {"$avg": "$confidence"}
                    }
                },
                {"$sort": {"count": -1}}
            ]
            
            crop_stats = []
            async for stat in self.collection.aggregate(pipeline):
                crop_stats.append({
                    "crop": stat["_id"],
                    "count": stat["count"],
                    "avg_confidence": round(stat["avg_confidence"], 4)
                })
            
            # Average nutrient values
            nutrient_pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "avg_nitrogen": {"$avg": "$nitrogen"},
                        "avg_phosphorus": {"$avg": "$phosphorus"},
                        "avg_potassium": {"$avg": "$potassium"}
                    }
                }
            ]
            
            nutrient_stats = {}
            async for stat in self.collection.aggregate(nutrient_pipeline):
                nutrient_stats = {
                    "avg_nitrogen": round(stat["avg_nitrogen"], 2),
                    "avg_phosphorus": round(stat["avg_phosphorus"], 2),
                    "avg_potassium": round(stat["avg_potassium"], 2)
                }
            
            return {
                "total_predictions": total_count,
                "predictions_by_crop": crop_stats,
                "average_nutrients": nutrient_stats
            }
        
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {}
    
    async def delete_all_predictions(self) -> int:
        """
        Delete all predictions from the database
        
        Returns:
            Number of deleted documents
        """
        try:
            result = await self.collection.delete_many({})
            logger.info(f"Deleted {result.deleted_count} predictions")
            return result.deleted_count
        
        except Exception as e:
            logger.error(f"Error deleting predictions: {str(e)}")
            return 0


# ============================================================================
# Synchronous Client (for testing/scripts)
# ============================================================================

class SyncMongoDBClient:
    """Synchronous MongoDB client for scripts and testing"""
    
    def __init__(self, mongodb_url: Optional[str] = None):
        """
        Initialize synchronous MongoDB client
        
        Args:
            mongodb_url: MongoDB connection URL
        """
        self.mongodb_url = mongodb_url or os.getenv(
            "MONGODB_URL",
            "mongodb://localhost:27017"
        )
        self.client = None
        self.database = None
        self.collection = None
    
    def connect(self, database_name: str = "crop_recommendation_db",
                collection_name: str = "predictions"):
        """
        Connect to MongoDB
        
        Args:
            database_name: Name of the database
            collection_name: Name of the collection
        """
        try:
            self.client = MongoClient(self.mongodb_url, serverSelectionTimeoutMS=5000)
            self.database = self.client[database_name]
            self.collection = self.database[collection_name]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"✓ Connected to MongoDB (sync)")
            
            return True
        
        except Exception as e:
            logger.error(f"✗ Failed to connect: {str(e)}")
            return False
    
    def disconnect(self):
        """Close connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed (sync)")
    
    def insert_prediction(self, nitrogen: float, phosphorus: float,
                         potassium: float, recommended_crop: str,
                         confidence: float) -> Optional[str]:
        """Insert a prediction record"""
        try:
            document = {
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "recommended_crop": recommended_crop,
                "confidence": confidence,
                "timestamp": datetime.utcnow()
            }
            
            result = self.collection.insert_one(document)
            return str(result.inserted_id)
        
        except Exception as e:
            logger.error(f"Error inserting: {str(e)}")
            return None


# ============================================================================
# Test Connection Function
# ============================================================================

async def test_connection():
    """Test MongoDB connection"""
    config = MongoDBConfig()
    success = await config.connect()
    
    if success:
        print("✓ MongoDB connection successful!")
        
        # Test insert
        db = PredictionDatabase(config)
        doc_id = await db.insert_prediction(
            nitrogen=90.0,
            phosphorus=42.0,
            potassium=43.0,
            recommended_crop="rice",
            confidence=0.92
        )
        
        if doc_id:
            print(f"✓ Test document inserted: {doc_id}")
        
        # Get statistics
        stats = await db.get_statistics()
        print(f"✓ Statistics: {stats}")
        
        await config.disconnect()
    else:
        print("✗ MongoDB connection failed!")


if __name__ == "__main__":
    import asyncio
    
    print("Testing MongoDB connection...")
    asyncio.run(test_connection())
