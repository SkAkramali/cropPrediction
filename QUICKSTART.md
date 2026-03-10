# 🚀 Quick Start Guide

Get the Crop Recommendation System running in **5 minutes**!

## Prerequisites Check

```powershell
# Verify you have everything installed
python --version    # Should be 3.8+
node --version      # Should be 16+
mongod --version    # Should be 5.0+
```

If anything is missing, install from:
- Python: https://www.python.org/downloads/
- Node.js: https://nodejs.org/
- MongoDB: https://www.mongodb.com/try/download/community

---

## Step 1: Backend Setup (3 minutes)

```powershell
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies (this may take 1-2 minutes)
pip install -r requirements.txt

# Train the model (this may take 30-60 seconds)
cd model
python train_model.py

# You should see "TRAINING COMPLETED SUCCESSFULLY!"
```

## Step 2: Start MongoDB (30 seconds)

```powershell
# Option 1: Start as service (recommended)
net start MongoDB

# Option 2: Start manually
mongod

# Keep this terminal open
```

## Step 3: Start Backend API (30 seconds)

```powershell
# Open NEW terminal
cd backend\api

# Activate virtual environment
..\venv\Scripts\activate

# Start API
python main.py

# You should see "✓ Model loaded successfully"
# Keep this terminal open
```

## Step 4: Start Frontend (1 minute)

```powershell
# Open NEW terminal
cd frontend

# Install dependencies (first time only, may take 1-2 minutes)
npm install

# Start development server
npm run dev

# Browser should open automatically to http://localhost:3000
```

---

## ✅ Verify Everything Works

1. **Check Backend API**: http://localhost:8000/docs
   - You should see interactive API documentation

2. **Check Frontend**: http://localhost:3000
   - You should see the Crop Recommendation interface

3. **Make a Test Prediction**:
   - Enter: N=90, P=42, K=43
   - Click "Predict Crop"
   - Should recommend "rice" with high confidence

---

## 🎯 Sample Test Values

Try these combinations:

| Crop | N | P | K | Expected Result |
|------|---|---|---|----------------|
| Rice | 90 | 42 | 43 | ~92% confidence |
| Wheat | 80 | 50 | 40 | ~88% confidence |
| Potato | 50 | 55 | 50 | ~90% confidence |
| Maize | 80 | 50 | 30 | ~85% confidence |

---

## 🔧 Troubleshooting

### "Module not found" error?
```powershell
# Reinstall requirements
pip install -r backend/requirements.txt
```

### "Cannot connect to server"?
```powershell
# Make sure backend is running
cd backend\api
python main.py
```

### "Model not loaded"?
```powershell
# Train the model first
cd backend\model
python train_model.py
```

### Port already in use?
```powershell
# Kill process using the port
# For port 8000:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## 📝 Summary

**Three terminals needed:**

1. **Terminal 1** - MongoDB
   ```powershell
   mongod
   ```

2. **Terminal 2** - Backend API
   ```powershell
   cd backend\api
   ..\venv\Scripts\activate
   python main.py
   ```

3. **Terminal 3** - Frontend
   ```powershell
   cd frontend
   npm run dev
   ```

**Access points:**
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- API: http://localhost:8000

---

## 🎉 You're Ready!

The system is now running. Try making some predictions!

For detailed documentation, see [README.md](README.md)

For API examples, see [API_EXAMPLES.txt](API_EXAMPLES.txt)

**Happy Farming! 🌾**
