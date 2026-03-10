# Crop Recommendation System

🌾 **AI-Powered Soil Analysis for Optimal Crop Selection**

A complete full-stack machine learning application that recommends the best crop to grow based on soil nutrient values (Nitrogen, Phosphorus, and Potassium).

---

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### Machine Learning
- **8 Algorithm Comparison**: Trains and compares multiple ML models
  - Random Forest, Decision Tree, K-Nearest Neighbors, Logistic Regression
  - XGBoost, Gradient Boosting, SVM, MLP Neural Network
- **Hyperparameter Tuning**: GridSearchCV on top-performing models
- **Best Model Auto-Selection**: Automatically selects the highest performing model
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Confusion Matrix & Feature Importance**: Detailed model analysis
- **Confidence Scoring**: Provides confidence scores for predictions
- **Model Persistence**: Saves trained models via joblib
- **Model Retraining**: Periodic retraining using stored prediction data

### Backend API
- **FastAPI Framework**: High-performance async API
- **RESTful Endpoints**: Clean and well-documented API
  - `POST /predict` — Single crop prediction
  - `POST /predict-batch` — CSV batch prediction
  - `GET /history` — Prediction history from MongoDB
  - `POST /retrain` — Trigger model retraining
- **Input Validation**: Pydantic models for request/response validation
- **Error Handling**: Comprehensive error messages
- **CORS Support**: Configured for frontend integration
- **Interactive Documentation**: Auto-generated Swagger UI and ReDoc

### Database
- **MongoDB Integration**: Stores all predictions with timestamps
- **History Tracking**: View past predictions
- **Statistics**: Query prediction patterns and trends
- **Async Operations**: Non-blocking database operations

### Frontend
- **Tab-Based Dashboard**: Modern React UI with three tabs
  - **Single Prediction**: NPK form with confidence visualization
  - **Bulk CSV Upload**: Upload CSV, view results table, download predictions
  - **Prediction History**: Browse all past predictions from MongoDB
- **Real-time Validation**: Client-side input validation
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern Styling**: Clean card-based layout with Inter font
- **Confidence Visualization**: Progress bars and color-coded badges

---

## 🏗️ System Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   React     │  HTTP   │   FastAPI    │  CRUD   │   MongoDB   │
│  Frontend   │ ──────> │   Backend    │ ──────> │  Database   │
│  (Port 3000)│ <────── │  (Port 8000) │ <────── │             │
└─────────────┘  JSON   └──────────────┘  Async  └─────────────┘
                              │
                              │ joblib
                              ▼
                        ┌──────────────┐
                        │   ML Model   │
                        │ (Random Forest)
                        └──────────────┘
```

### Data Flow

1. **User Input** → User enters N, P, K values in React frontend
2. **API Request** → Frontend sends POST request to `/predict` endpoint
3. **Prediction** → Backend loads model and makes prediction
4. **Database Storage** → Prediction saved to MongoDB
5. **Response** → Result sent back to frontend with confidence score
6. **Display** → Frontend displays recommended crop and details

---

## 🛠️ Technology Stack

### Machine Learning
- **Python 3.8+**
- **scikit-learn**: ML algorithms and preprocessing
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **joblib**: Model serialization

### Backend
- **FastAPI**: Modern async web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **Motor**: Async MongoDB driver

### Database
- **MongoDB**: NoSQL document database

### Frontend
- **React 18**: UI library
- **Vite**: Build tool and dev server
- **Axios**: HTTP client
- **CSS3**: Styling

---

## 📁 Project Structure

```
crop-recomendation-engine/
│
├── backend/
│   ├── model/
│   │   ├── train_model.py          # ML training script (8 models)
│   │   ├── retrain_model.py        # Retraining from MongoDB data
│   │   └── Crop_recommendation.csv # Dataset
│   ├── saved_model/
│   │   ├── crop_model.pkl          # Trained model (generated)
│   │   ├── scaler.pkl              # Feature scaler (generated)
│   │   ├── label_encoder.pkl       # Label encoder (generated)
│   │   └── model_metadata.json     # Model info (generated)
│   ├── api/
│   │   └── main.py                 # FastAPI application
│   ├── database/
│   │   └── mongodb_config.py       # MongoDB configuration
│   └── requirements.txt            # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── PredictTab.jsx      # Single prediction form
│   │   │   ├── UploadCSVTab.jsx    # Bulk CSV upload & results
│   │   │   ├── HistoryTab.jsx      # Prediction history table
│   │   │   └── Navbar.jsx          # Navigation bar with tabs
│   │   ├── App.jsx                 # Root component (tab router)
│   │   ├── App.css                 # Dashboard styles
│   │   ├── index.css               # Global styles
│   │   └── main.jsx                # Entry point
│   ├── index.html                  # HTML template
│   ├── package.json                # Node dependencies
│   └── vite.config.js              # Vite configuration
│
├── API_EXAMPLES.txt                # API testing examples
├── QUICKSTART.md                   # Quick start guide
└── README.md                       # This file
```

---

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

### Required Software
- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **Node.js 16+** ([Download](https://nodejs.org/))
- **MongoDB 5.0+** ([Download](https://www.mongodb.com/try/download/community))
- **Git** (optional, for cloning)

### Verify Installations

```powershell
# Check Python version
python --version

# Check Node.js version
node --version

# Check npm version
npm --version

# Check MongoDB status
mongod --version
```

---

## 🚀 Installation & Setup

### Step 1: Clone or Download the Project

```powershell
# If using Git
git clone <repository-url>
cd "crop recomendation engine"

# Or download and extract the ZIP file
```

### Step 2: Backend Setup

#### 2.1 Create Python Virtual Environment

```powershell
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Your prompt should now show (venv)
```

#### 2.2 Install Python Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# This will install:
# - numpy, pandas, scikit-learn (ML libraries)
# - fastapi, uvicorn (API framework)
# - motor, pymongo (MongoDB drivers)
# - and other dependencies
```

#### 2.3 Configure Environment Variables

```powershell
# Copy the example .env file
copy .env.example .env

# Edit .env if needed (default values should work for local development)
```

#### 2.4 Train the Machine Learning Model

```powershell
# Navigate to model directory
cd model

# Run training script
python train_model.py

# This will:
# 1. Create sample dataset (if not exists)
# 2. Train 4 different ML models
# 3. Compare their accuracies
# 4. Save the best model
# 
# Expected output:
# - Dataset with 1000 samples created
# - Models trained and compared
# - Best model saved to saved_model/
```

Expected training output:
```
═══════════════════════════════════════════════════════════
               CROP RECOMMENDATION MODEL
                    TRAINING PIPELINE
═══════════════════════════════════════════════════════════

LOADING DATASET
───────────────────────────────────────────────────────────
Dataset shape: (1000, 4)
...

MODEL COMPARISON
───────────────────────────────────────────────────────────
Model                     Accuracy        CV Score
───────────────────────────────────────────────────────────
Random Forest             0.9850 (98.50%)  0.9832 (+/- 0.0145)
Decision Tree             0.9400 (94.00%)  0.9356 (+/- 0.0234)
KNN                       0.9200 (92.00%)  0.9123 (+/- 0.0289)
Logistic Regression       0.8950 (89.50%)  0.8934 (+/- 0.0312)

═══════════════════════════════════════════════════════════
BEST MODEL: Random Forest
Accuracy: 0.9850
═══════════════════════════════════════════════════════════
```

### Step 3: Start MongoDB

#### 3.1 Start MongoDB Service

```powershell
# Start MongoDB (Windows)
# Option 1: Start as Windows Service
net start MongoDB

# Option 2: Start manually
mongod

# MongoDB should now be running on mongodb://localhost:27017
```

#### 3.2 Verify MongoDB Connection (Optional)

```powershell
# Open MongoDB shell
mongo

# Or use MongoDB Compass GUI
```

### Step 4: Start Backend API

```powershell
# Navigate to api directory (if not already there)
cd ..\api

# Start FastAPI server
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Server will start on http://localhost:8000
```

Expected output:
```
INFO:     Starting Crop Recommendation API...
INFO:     ✓ Model loaded successfully
INFO:     ✓ Scaler loaded successfully
INFO:     ✓ Model metadata loaded: Random Forest
INFO:     ✓ Connected to MongoDB at mongodb://localhost:27017
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Test the API:**
- Open browser: http://localhost:8000/docs
- You should see interactive API documentation

### Step 5: Frontend Setup

#### 5.1 Install Node Dependencies

```powershell
# Open a NEW PowerShell window
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# This will install React, Vite, Axios, and other packages
```

#### 5.2 Configure Frontend Environment

The `.env` file should already exist with:
```
VITE_API_URL=http://localhost:8000
```

If the backend is running on a different port, update this file.

#### 5.3 Start Frontend Development Server

```powershell
# Start Vite dev server
npm run dev

# Frontend will start on http://localhost:3000
# Browser should open automatically
```

Expected output:
```
  VITE v5.0.8  ready in 523 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

---

## 🎯 Usage

### Making Predictions via Web Interface

1. **Open Frontend**: Navigate to http://localhost:3000
2. **View Model Info**: See the loaded model details at the top
3. **Enter Values**: Input Nitrogen (N), Phosphorus (P), and Potassium (K) values
   - Example for Rice: N=90, P=42, K=43
   - Example for Wheat: N=80, P=50, K=40
   - Example for Potato: N=50, P=55, K=50
4. **Click "Predict Crop"**: Submit the form
5. **View Results**: See the recommended crop and confidence score
6. **Check History**: View recent predictions by clicking "Show History"

### Making Predictions via API

#### Using curl (PowerShell)

```powershell
# Make a prediction
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"nitrogen\": 90, \"phosphorus\": 42, \"potassium\": 43}'
```

#### Using Python

```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={
        'nitrogen': 90,
        'phosphorus': 42,
        'potassium': 43
    }
)

result = response.json()
print(f"Recommended Crop: {result['prediction']['recommended_crop']}")
print(f"Confidence: {result['prediction']['confidence_score'] * 100:.2f}%")
```

#### Using JavaScript

```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    nitrogen: 90,
    phosphorus: 42,
    potassium: 43
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

More examples available in `API_EXAMPLES.txt`.

---

## 📚 API Documentation

### Endpoints

#### 1. Health Check
```
GET /health
```
Check API status and connectivity.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true,
  "timestamp": "2026-03-10T12:00:00"
}
```

#### 2. Get Model Information
```
GET /model-info
```
Get details about the loaded ML model.

**Response:**
```json
{
  "model_name": "Random Forest",
  "features": ["N", "P", "K"],
  "classes": ["rice", "wheat", "maize", "cotton", ...],
  "accuracy": 0.9850,
  "trained_date": "2026-03-10T12:00:00"
}
```

#### 3. Predict Crop
```
POST /predict
```
Get crop recommendation based on soil nutrients.

**Request Body:**
```json
{
  "nitrogen": 90,
  "phosphorus": 42,
  "potassium": 43
}
```

**Response:**
```json
{
  "success": true,
  "input": {
    "nitrogen": 90,
    "phosphorus": 42,
    "potassium": 43
  },
  "prediction": {
    "recommended_crop": "rice",
    "confidence_score": 0.9234
  },
  "timestamp": "2026-03-10T12:00:00"
}
```

#### 4. Get Prediction History
```
GET /history?limit=10
```
Retrieve recent predictions from database.

**Response:**
```json
{
  "success": true,
  "count": 10,
  "data": [
    {
      "_id": "65abc123...",
      "nitrogen": 90,
      "phosphorus": 42,
      "potassium": 43,
      "recommended_crop": "rice",
      "confidence": 0.92,
      "timestamp": "2026-03-10T12:00:00"
    }
  ]
}
```

#### 5. Clear History
```
DELETE /history
```
Delete all prediction records.

**Response:**
```json
{
  "success": true,
  "message": "Deleted 10 records",
  "deleted_count": 10
}
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📊 Model Performance

### Algorithm Comparison

The system automatically trains and compares four classification algorithms:

| Algorithm | Typical Accuracy | Training Time | Pros | Cons |
|-----------|-----------------|---------------|------|------|
| **Random Forest** | 98-99% | Medium | High accuracy, handles non-linear data | Slower prediction |
| **Decision Tree** | 94-96% | Fast | Fast, interpretable | May overfit |
| **KNN** | 92-94% | Fast | Simple, no training phase | Slow with large datasets |
| **Logistic Regression** | 89-91% | Fast | Fast, probabilistic | Assumes linear relationships |

### Model Selection

The system automatically selects the best performing model based on:
1. **Test Set Accuracy**: Performance on unseen data
2. **Cross-Validation Score**: 5-fold CV for robustness
3. **Consistency**: Low standard deviation in CV scores

### Sample Predictions

| Crop | Nitrogen (N) | Phosphorus (P) | Potassium (K) | Confidence |
|------|-------------|----------------|---------------|------------|
| Rice | 80-100 | 35-50 | 35-50 | 92-98% |
| Wheat | 70-90 | 40-60 | 30-50 | 88-95% |
| Maize | 70-90 | 40-60 | 20-40 | 85-92% |
| Potato | 40-60 | 45-65 | 40-60 | 90-96% |
| Cotton | 100-130 | 35-55 | 15-35 | 87-94% |

---

## 🖼️ Screenshots

### Home Page
![Home Page](docs/screenshots/home.png)
*Main interface with input form*

### Prediction Result
![Prediction Result](docs/screenshots/result.png)
*Crop recommendation with confidence score*

### API Documentation
![API Docs](docs/screenshots/api-docs.png)
*Interactive Swagger UI documentation*

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Module not found" Error

**Problem**: Python modules not installed

**Solution**:
```powershell
# Activate virtual environment
cd backend
.\venv\Scripts\activate

# Reinstall requirements
pip install -r requirements.txt
```

#### 2. "Cannot connect to server" Error

**Problem**: Backend API not running

**Solution**:
```powershell
# Start the backend
cd backend\api
python main.py

# Verify it's running: http://localhost:8000/health
```

#### 3. "Database not connected" Warning

**Problem**: MongoDB not running

**Solution**:
```powershell
# Start MongoDB service
net start MongoDB

# Or start manually
mongod
```

#### 4. "Model not loaded" Error

**Problem**: Model files not generated

**Solution**:
```powershell
# Train the model first
cd backend\model
python train_model.py

# Check that files were created in saved_model/
```

#### 5. Frontend Shows CORS Error

**Problem**: CORS not configured properly

**Solution**:
- Ensure backend `.env` has correct ALLOWED_ORIGINS
- Restart the backend API
- Clear browser cache

#### 6. Port Already in Use

**Problem**: Port 8000 or 3000 already occupied

**Solution**:
```powershell
# Change backend port
uvicorn main:app --port 8001

# Change frontend port
# Edit vite.config.js and change port to 3001
```

### Logs and Debugging

#### Backend Logs
```powershell
# Run with verbose logging
uvicorn main:app --log-level debug
```

#### Frontend Logs
- Open browser Developer Tools (F12)
- Check Console tab for errors
- Check Network tab for API requests

#### Database Logs
```powershell
# Connect to MongoDB
mongo

# Check database
use crop_recommendation_db
db.predictions.find().pretty()
```

---

## 🚢 Production Deployment

### Backend Deployment

#### Docker (Recommended)

```dockerfile
# Create Dockerfile in backend/
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```powershell
# Build and run
docker build -t crop-api .
docker run -p 8000:8000 crop-api
```

#### Cloud Platforms
- **Heroku**: Use Procfile
- **AWS**: Deploy on EC2 or ECS
- **Google Cloud**: Use Cloud Run
- **Azure**: Deploy on App Service

### Frontend Deployment

```powershell
# Build for production
npm run build

# Deploy the dist/ folder to:
# - Netlify
# - Vercel
# - GitHub Pages
# - AWS S3 + CloudFront
```

### Database Deployment

- **MongoDB Atlas**: Managed MongoDB in the cloud
- **AWS DocumentDB**: AWS-managed MongoDB-compatible database
- **Self-hosted**: Deploy MongoDB on your server

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use ESLint for JavaScript code
- Write meaningful commit messages
- Add tests for new features
- Update documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

Created by **AI Assistant**

- Built with ❤️ using Python, FastAPI, React, and MongoDB
- Date: March 2026

---

## 🙏 Acknowledgments

- **scikit-learn** for ML algorithms
- **FastAPI** for the excellent framework
- **React** team for the UI library
- **MongoDB** for the database solution
- Agricultural research community for domain knowledge

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [API Documentation](#api-documentation)
3. Check existing issues on GitHub
4. Create a new issue with:
   - Detailed description
   - Steps to reproduce
   - Error messages
   - System information

---

## 🔄 Version History

### Version 1.0.0 (March 2026)
- Initial release
- 4 ML algorithms implemented
- FastAPI backend with full CRUD
- React frontend with modern UI
- MongoDB integration
- Comprehensive documentation

---

## 🎓 Educational Value

This project demonstrates:

- **Full-Stack Development**: Backend + Frontend + Database
- **Machine Learning**: Supervised learning for classification
- **API Design**: RESTful API with proper structure
- **Modern Web Development**: React hooks, async/await, responsive design
- **Database Operations**: CRUD operations with MongoDB
- **DevOps**: Environment configuration, deployment preparation

Perfect for learning:
- Machine Learning implementation
- Building production-ready APIs
- Full-stack application development
- Database integration
- Modern frontend development

---

**Happy Farming! 🌾🚜**
