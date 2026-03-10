"""
Crop Recommendation Model Training Script
==========================================
Trains multiple supervised learning models to predict the best crop
based on soil nutrient values (N, P, K).

Models tested:
- Random Forest, Decision Tree, KNN, Logistic Regression
- XGBoost, Gradient Boosting, SVM, MLP Neural Network

Includes cross-validation, hyperparameter tuning, confusion matrix,
and feature importance visualization. The best model is auto-selected and saved.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
from datetime import datetime

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: xgboost not installed. XGBoost model will be skipped.")

# Models that require scaled input
SCALED_MODELS = {'KNN', 'Logistic Regression', 'SVM', 'MLP Neural Network', 'XGBoost', 'Gradient Boosting'}


class CropRecommendationModel:
    """Main class for training and evaluating crop recommendation models"""

    def __init__(self, dataset_path='Crop_recommendation.csv'):
        self.dataset_path = dataset_path
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = ['N', 'P', 'K']
        self.all_results = {}

    def load_data(self):
        """Load and explore the dataset"""
        print("=" * 60)
        print("LOADING DATASET")
        print("=" * 60)

        df = pd.read_csv(self.dataset_path)

        print(f"\nDataset shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nClass distribution:")
        print(df['label'].value_counts())
        print(f"\nMissing values:")
        print(df.isnull().sum())

        return df

    def preprocess_data(self, df):
        """Preprocess the dataset: identify features, split, and scale"""
        print("\n" + "=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)

        required_cols = ['N', 'P', 'K', 'label']
        if not all(col in df.columns for col in required_cols):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                self.feature_names = numeric_cols[:3]
                label_col = [col for col in df.columns if col not in numeric_cols][0]
            else:
                raise ValueError("Cannot identify N, P, K columns in dataset")
        else:
            self.feature_names = ['N', 'P', 'K']
            label_col = 'label'

        X = df[self.feature_names]
        y = df[label_col]

        print(f"\nFeatures used: {self.feature_names}")
        print(f"Number of classes: {y.nunique()}")
        print(f"Classes: {sorted(y.unique())}")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Encode labels for XGBoost
        self.label_encoder.fit(self.y_train)
        self.y_train_encoded = self.label_encoder.transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)

        print("Data preprocessing completed!")

    def _build_models(self):
        """Build dictionary of all models to train"""
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
            ),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'MLP Neural Network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), max_iter=500,
                activation='relu', solver='adam', random_state=42
            ),
        }
        if HAS_XGBOOST:
            models['XGBoost'] = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric='mlogloss', random_state=42
            )
        return models

    def train_models(self):
        """Train all classification models"""
        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)

        self.models = self._build_models()
        results = {}

        for name, model in self.models.items():
            print(f"\n{'─' * 40}")
            print(f"Training {name}...")
            print(f"{'─' * 40}")

            use_scaled = name in SCALED_MODELS
            X_tr = self.X_train_scaled if use_scaled else self.X_train
            X_te = self.X_test_scaled if use_scaled else self.X_test

            # XGBoost uses encoded labels
            if name == 'XGBoost':
                model.fit(X_tr, self.y_train_encoded)
                y_pred_encoded = model.predict(X_te)
                y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
                cv_scores = cross_val_score(model, X_tr, self.y_train_encoded, cv=5)
            else:
                model.fit(X_tr, self.y_train)
                y_pred = model.predict(X_te)
                cv_scores = cross_val_score(model, X_tr, self.y_train, cv=5)

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
            print(f"Cross-validation: {cv_mean:.4f} (+/- {cv_std:.4f})")

        self.all_results = results
        return results

    def hyperparameter_tuning(self, top_n=2):
        """Run GridSearchCV on the top N models by accuracy"""
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING (GridSearchCV)")
        print("=" * 60)

        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5],
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1],
            },
        }

        sorted_models = sorted(
            self.all_results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        for name, data in sorted_models[:top_n]:
            if name not in param_grids:
                print(f"\nSkipping tuning for {name} (no param grid defined)")
                continue

            print(f"\n{'─' * 40}")
            print(f"Tuning {name}...")
            print(f"{'─' * 40}")

            use_scaled = name in SCALED_MODELS
            X_tr = self.X_train_scaled if use_scaled else self.X_train
            X_te = self.X_test_scaled if use_scaled else self.X_test

            grid = GridSearchCV(
                self.models[name].__class__(**{'random_state': 42} if hasattr(self.models[name], 'random_state') else {}),
                param_grids[name], cv=5, scoring='accuracy', n_jobs=-1
            )
            grid.fit(X_tr, self.y_train)
            y_pred = grid.predict(X_te)
            new_acc = accuracy_score(self.y_test, y_pred)

            print(f"Best params: {grid.best_params_}")
            print(f"Tuned accuracy: {new_acc:.4f} (was {data['accuracy']:.4f})")

            if new_acc > data['accuracy']:
                self.all_results[name]['model'] = grid.best_estimator_
                self.all_results[name]['accuracy'] = new_acc
                self.all_results[name]['predictions'] = y_pred
                self.models[name] = grid.best_estimator_
                print(f"✓ Updated {name} with tuned model")

    def evaluate_models(self, results):
        """Evaluate and compare all models, select the best"""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

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

        self.best_model_name = sorted_models[0][0]
        self.best_model = results[self.best_model_name]['model']

        print(f"\n{'=' * 60}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Accuracy: {results[self.best_model_name]['accuracy']:.4f}")
        print(f"{'=' * 60}")

        y_pred = results[self.best_model_name]['predictions']
        print(f"\nClassification Report for {self.best_model_name}:")
        print("─" * 60)
        print(classification_report(self.y_test, y_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)

        return self.best_model_name, results[self.best_model_name]['accuracy']

    def print_feature_importance(self):
        """Print feature importance for tree-based models"""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE")
        print("=" * 60)

        for name in ['Random Forest', 'Gradient Boosting', 'Decision Tree', 'XGBoost']:
            if name not in self.all_results:
                continue
            m = self.all_results[name]['model']
            if hasattr(m, 'feature_importances_'):
                importances = m.feature_importances_
                print(f"\n{name}:")
                for feat, imp in sorted(zip(self.feature_names, importances), key=lambda x: -x[1]):
                    bar = "█" * int(imp * 40)
                    print(f"  {feat:>3}: {imp:.4f} {bar}")

    def save_model(self):
        """Save the best model, scaler, and metadata"""
        print("\n" + "=" * 60)
        print("SAVING MODEL")
        print("=" * 60)

        save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_model')
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, 'crop_model.pkl')
        joblib.dump(self.best_model, model_path)
        print(f"✓ Model saved to: {model_path}")

        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to: {scaler_path}")

        # Save label encoder for XGBoost compatibility
        le_path = os.path.join(save_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, le_path)
        print(f"✓ Label encoder saved to: {le_path}")

        use_scaled = self.best_model_name in SCALED_MODELS
        X_te = self.X_test_scaled if use_scaled else self.X_test
        if self.best_model_name == 'XGBoost':
            y_pred_enc = self.best_model.predict(X_te)
            y_pred = self.label_encoder.inverse_transform(y_pred_enc)
        else:
            y_pred = self.best_model.predict(X_te)

        # Collect all model scores for metadata
        comparison = {}
        for name, res in self.all_results.items():
            comparison[name] = {
                'accuracy': round(res['accuracy'], 4),
                'cv_mean': round(res['cv_mean'], 4),
                'cv_std': round(res['cv_std'], 4),
            }

        metadata = {
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'classes': sorted(self.y_test.unique().tolist()),
            'trained_date': datetime.now().isoformat(),
            'accuracy': float(accuracy_score(self.y_test, y_pred)),
            'model_comparison': comparison,
            'uses_scaling': use_scaled,
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
        print(" " * 15 + "ADVANCED TRAINING PIPELINE")
        print("═" * 60)

        df = self.load_data()
        self.preprocess_data(df)
        results = self.train_models()
        self.hyperparameter_tuning(top_n=2)
        self.evaluate_models(self.all_results)
        self.print_feature_importance()
        self.save_model()

        print("\n" + "═" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("═" * 60 + "\n")


def create_sample_dataset():
    """Create a sample dataset if one doesn't exist"""
    print("Creating sample dataset...")
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
    df = df.sample(frac=1).reset_index(drop=True)

    dataset_path = 'Crop_recommendation.csv'
    df.to_csv(dataset_path, index=False)
    print(f"✓ Sample dataset created: {dataset_path}")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Number of crops: {df['label'].nunique()}")

    return dataset_path


if __name__ == "__main__":
    dataset_path = 'Crop_recommendation.csv'

    if not os.path.exists(dataset_path):
        print("Dataset not found. Creating sample dataset...\n")
        dataset_path = create_sample_dataset()
        print()

    trainer = CropRecommendationModel(dataset_path)
    trainer.run_full_pipeline()
