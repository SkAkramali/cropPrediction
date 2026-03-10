"""
Model Retraining Script
=======================
Pulls historical prediction data from MongoDB, merges with original dataset,
and retrains the model for improved accuracy.

Usage:
    python retrain_model.py
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Add parent dir so we can import train_model
sys.path.insert(0, os.path.dirname(__file__))
from train_model import CropRecommendationModel


MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "crop_recommendation_db"
COLLECTION_NAME = "predictions"
MIN_CONFIDENCE = 0.80  # Only use predictions above this confidence for retraining


def fetch_historical_data():
    """Pull historical prediction data from MongoDB"""
    print("=" * 60)
    print("FETCHING HISTORICAL DATA FROM MONGODB")
    print("=" * 60)

    try:
        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print(f"✓ Connected to MongoDB at {MONGODB_URL}")
    except ConnectionFailure as e:
        print(f"✗ Could not connect to MongoDB: {e}")
        return pd.DataFrame()

    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Only use high-confidence predictions as pseudo-labels
    cursor = collection.find(
        {"confidence": {"$gte": MIN_CONFIDENCE}},
        {"_id": 0, "nitrogen": 1, "phosphorus": 1, "potassium": 1, "recommended_crop": 1, "confidence": 1}
    )

    records = list(cursor)
    client.close()

    if not records:
        print("No historical data found above confidence threshold.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.rename(columns={
        'nitrogen': 'N',
        'phosphorus': 'P',
        'potassium': 'K',
        'recommended_crop': 'label'
    })
    df = df[['N', 'P', 'K', 'label']]

    print(f"✓ Fetched {len(df)} historical predictions")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def load_original_dataset(dataset_path):
    """Load the original training dataset"""
    if not os.path.exists(dataset_path):
        print(f"✗ Original dataset not found at {dataset_path}")
        return pd.DataFrame()

    df = pd.read_csv(dataset_path)
    print(f"✓ Original dataset loaded: {len(df)} rows")
    return df


def retrain():
    """Main retraining pipeline"""
    print("\n" + "═" * 60)
    print(" " * 10 + "CROP RECOMMENDATION MODEL RETRAINING")
    print("═" * 60)

    model_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(model_dir, 'Crop_recommendation.csv')

    # Step 1: Load original dataset
    original_df = load_original_dataset(dataset_path)

    # Step 2: Fetch historical data from MongoDB
    historical_df = fetch_historical_data()

    # Step 3: Merge datasets
    if historical_df.empty:
        print("\nNo historical data to merge. Retraining on original dataset only.")
        merged_df = original_df
    else:
        merged_df = pd.concat([original_df, historical_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['N', 'P', 'K', 'label'])
        print(f"\n✓ Merged dataset: {len(merged_df)} rows "
              f"(original: {len(original_df)}, historical: {len(historical_df)})")

    if merged_df.empty:
        print("✗ No data available for retraining.")
        return False

    # Step 4: Save the merged dataset as backup
    backup_path = os.path.join(model_dir, f'merged_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    merged_df.to_csv(backup_path, index=False)
    print(f"✓ Merged dataset saved to: {backup_path}")

    # Step 5: Retrain model using the full pipeline
    print("\n" + "=" * 60)
    print("RETRAINING MODEL")
    print("=" * 60)

    trainer = CropRecommendationModel(dataset_path=backup_path)
    trainer.run_full_pipeline()

    # Step 6: Log retraining event
    log_path = os.path.join(model_dir, '..', 'saved_model', 'retrain_log.json')
    log_entry = {
        'retrained_at': datetime.now().isoformat(),
        'original_rows': len(original_df),
        'historical_rows': len(historical_df),
        'merged_rows': len(merged_df),
        'min_confidence_threshold': MIN_CONFIDENCE,
    }

    log_data = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)

    log_data.append(log_entry)
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)

    print(f"\n✓ Retraining log updated: {log_path}")
    print("\n" + "═" * 60)
    print("RETRAINING COMPLETED SUCCESSFULLY!")
    print("═" * 60 + "\n")
    return True


if __name__ == "__main__":
    retrain()
