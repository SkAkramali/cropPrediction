"""
Crop Recommendation Model Training Script
==========================================
This script trains multiple supervised learning models to predict the best crop
based on soil nutrient values (N, P, K).

Models tested:
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)
- Logistic Regression

The best performing model is saved for production use.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
from datetime import datetime


class CropRecommendationModel:
    """Main class for training and evaluating crop recommendation models"""
    
    def __init__(self, dataset_path='Crop_recommendation.csv'):
        """
        Initialize the model trainer
        
        Args:
            dataset_path (str): Path to the dataset CSV file
        """
        self.dataset_path = dataset_path
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = ['N', 'P', 'K']
        
    def load_data(self):
        """Load and explore the dataset"""
        print("=" * 60)
        print("LOADING DATASET")
        print("=" * 60)
        
        # Load dataset
        df = pd.read_csv(self.dataset_path)
        
        print(f"\nDataset shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        print(f"\nDataset info:")
        print(df.info())
        
        print(f"\nClass distribution:")
        print(df['label'].value_counts())
        
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset
        
        Args:
            df (DataFrame): Raw dataset
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)
        
        # Check if dataset has all required columns
        required_cols = ['N', 'P', 'K', 'label']
        if not all(col in df.columns for col in required_cols):
            print("\nWarning: Dataset doesn't have standard column names.")
            print(f"Available columns: {df.columns.tolist()}")
            
            # Try to identify columns (common alternatives)
            # Assuming first 3 numeric columns are N, P, K and last is label
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                self.feature_names = numeric_cols[:3]
                label_col = [col for col in df.columns if col not in numeric_cols][0]
            else:
                raise ValueError("Cannot identify N, P, K columns in dataset")
        else:
            self.feature_names = ['N', 'P', 'K']
            label_col = 'label'
        
        # Extract features and labels
        X = df[self.feature_names]
        y = df[label_col]
        
        print(f"\nFeatures used: {self.feature_names}")
        print(f"Number of classes: {y.nunique()}")
        print(f"Classes: {sorted(y.unique())}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\nData preprocessing completed!")
        
    def train_models(self):
        """Train multiple classification models"""
        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)
        
        # Define models
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'─' * 40}")
            print(f"Training {name}...")
            print(f"{'─' * 40}")
            
            # Train model
            if name in ['KNN', 'Logistic Regression']:
                # These models benefit from scaled features
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, self.X_train_scaled, self.y_train, cv=5
                )
            else:
                # Tree-based models don't require scaling
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train, cv=5
                )
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred
            }
            
            print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"Cross-validation score: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        return results
    
    def evaluate_models(self, results):
        """
        Evaluate and compare all models
        
        Args:
            results (dict): Dictionary containing results from all models
        """
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        # Sort models by accuracy
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        print("\n{:<25} {:<15} {:<20}".format("Model", "Accuracy", "CV Score"))
        print("─" * 60)
        
        for name, metrics in sorted_models:
            print("{:<25} {:.4f} ({:.2f}%)  {:.4f} (+/- {:.4f})".format(
                name,
                metrics['accuracy'],
                metrics['accuracy'] * 100,
                metrics['cv_mean'],
                metrics['cv_std']
            ))
        
        # Select best model
        self.best_model_name = sorted_models[0][0]
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\n{'=' * 60}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Accuracy: {results[self.best_model_name]['accuracy']:.4f}")
        print(f"{'=' * 60}")
        
        # Detailed evaluation of best model
        print(f"\nDetailed Classification Report for {self.best_model_name}:")
        print("─" * 60)
        
        y_pred = results[self.best_model_name]['predictions']
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        return self.best_model_name, results[self.best_model_name]['accuracy']
    
    def save_model(self):
        """Save the best model and scaler"""
        print("\n" + "=" * 60)
        print("SAVING MODEL")
        print("=" * 60)
        
        # Create directory if it doesn't exist
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_model')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, 'crop_model.pkl')
        joblib.dump(self.best_model, model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to: {scaler_path}")
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'classes': sorted(self.y_test.unique().tolist()),
            'trained_date': datetime.now().isoformat(),
            'accuracy': float(accuracy_score(
                self.y_test,
                self.best_model.predict(
                    self.X_test_scaled if self.best_model_name in ['KNN', 'Logistic Regression']
                    else self.X_test
                )
            ))
        }
        
        metadata_path = os.path.join(save_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Metadata saved to: {metadata_path}")
        
        print("\nModel training and saving completed successfully!")
    
    def run_full_pipeline(self):
        """Execute the complete training pipeline"""
        print("\n" + "═" * 60)
        print(" " * 15 + "CROP RECOMMENDATION MODEL")
        print(" " * 20 + "TRAINING PIPELINE")
        print("═" * 60)
        
        # Load data
        df = self.load_data()
        
        # Preprocess
        self.preprocess_data(df)
        
        # Train models
        results = self.train_models()
        
        # Evaluate
        self.evaluate_models(results)
        
        # Save best model
        self.save_model()
        
        print("\n" + "═" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("═" * 60 + "\n")


def create_sample_dataset():
    """
    Create a sample dataset if one doesn't exist
    This is useful for testing purposes
    """
    print("Creating sample dataset...")
    
    # Sample data for common crops
    np.random.seed(42)
    
    crops_data = {
        'rice': {'N': (80, 100), 'P': (35, 50), 'K': (35, 50)},
        'wheat': {'N': (70, 90), 'P': (40, 60), 'K': (30, 50)},
        'maize': {'N': (70, 90), 'P': (40, 60), 'K': (20, 40)},
        'cotton': {'N': (100, 130), 'P': (35, 55), 'K': (15, 35)},
        'sugarcane': {'N': (90, 120), 'P': (30, 50), 'K': (30, 50)},
        'potato': {'N': (40, 60), 'P': (45, 65), 'K': (40, 60)},
        'tomato': {'N': (30, 50), 'P': (50, 70), 'K': (50, 70)},
        'coffee': {'N': (90, 110), 'P': (20, 40), 'K': (25, 45)},
        'apple': {'N': (10, 30), 'P': (120, 140), 'K': (190, 210)},
        'banana': {'N': (90, 110), 'P': (70, 90), 'K': (40, 60)},
    }
    
    data = []
    samples_per_crop = 100
    
    for crop, ranges in crops_data.items():
        for _ in range(samples_per_crop):
            n = np.random.uniform(ranges['N'][0], ranges['N'][1])
            p = np.random.uniform(ranges['P'][0], ranges['P'][1])
            k = np.random.uniform(ranges['K'][0], ranges['K'][1])
            
            data.append({
                'N': round(n, 2),
                'P': round(p, 2),
                'K': round(k, 2),
                'label': crop
            })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    dataset_path = 'Crop_recommendation.csv'
    df.to_csv(dataset_path, index=False)
    print(f"✓ Sample dataset created: {dataset_path}")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Number of crops: {df['label'].nunique()}")
    
    return dataset_path


if __name__ == "__main__":
    # Check if dataset exists, if not create a sample one
    dataset_path = 'Crop_recommendation.csv'
    
    if not os.path.exists(dataset_path):
        print("Dataset not found. Creating sample dataset...\n")
        dataset_path = create_sample_dataset()
        print()
    
    # Initialize and run training pipeline
    trainer = CropRecommendationModel(dataset_path)
    trainer.run_full_pipeline()
