# Crop Recommendation Backend

FastAPI backend for the crop recommendation system.

## Prerequisites

- Python 3.10+
- MongoDB (running locally or remote)

## Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure MongoDB** (optional)
   
   Edit `database/mongodb_config.py` if needed to set your MongoDB connection string.

## Running the Server

```bash
uvicorn api.main:app --reload
```

The API will be available at: `http://localhost:8000`

## API Endpoints

| Method | Endpoint         | Description                    |
|--------|------------------|--------------------------------|
| POST   | `/predict`       | Single crop prediction         |
| POST   | `/predict-batch` | CSV file batch prediction      |
| GET    | `/history`       | Prediction history from MongoDB|
| POST   | `/retrain`       | Trigger model retraining       |
| GET    | `/health`        | Health check                   |
| GET    | `/model-info`    | Loaded model metadata          |

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
