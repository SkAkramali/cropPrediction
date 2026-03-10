# Project Dependencies Summary

## Backend Dependencies

### Core ML Libraries
```
numpy==1.24.3                    # Numerical computing
pandas==2.0.3                    # Data manipulation
scikit-learn==1.3.0              # Machine learning algorithms
joblib==1.3.2                    # Model serialization
```

### API Framework
```
fastapi==0.104.1                 # Modern web framework
uvicorn[standard]==0.24.0        # ASGI server
pydantic==2.5.0                  # Data validation
python-multipart==0.0.6          # Form data support
```

### Database
```
motor==3.3.2                     # Async MongoDB driver
pymongo==4.6.1                   # Sync MongoDB driver
```

### Utilities
```
python-dotenv==1.0.0             # Environment variables
python-jose[cryptography]==3.3.0 # JWT tokens (optional)
passlib[bcrypt]==1.7.4           # Password hashing (optional)
```

### Testing (Optional)
```
pytest==7.4.3                    # Testing framework
httpx==0.25.2                    # Async HTTP client
pytest-asyncio==0.21.1           # Async test support
```

**Total Backend Size**: ~150 MB

---

## Frontend Dependencies

### Core
```
react: ^18.2.0                   # UI library
react-dom: ^18.2.0               # React DOM renderer
axios: ^1.6.2                    # HTTP client
```

### Development
```
@vitejs/plugin-react: ^4.2.1     # Vite React plugin
vite: ^5.0.8                     # Build tool
eslint: ^8.55.0                  # Code linter
```

**Total Frontend Size**: ~200 MB (node_modules)

---

## System Requirements

### Minimum
- **OS**: Windows 10, macOS 10.15, Ubuntu 20.04
- **CPU**: Dual-core 2.0 GHz
- **RAM**: 4 GB
- **Storage**: 2 GB free space
- **Python**: 3.8+
- **Node.js**: 16+
- **MongoDB**: 5.0+

### Recommended
- **OS**: Windows 11, macOS 12+, Ubuntu 22.04
- **CPU**: Quad-core 3.0 GHz
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **Python**: 3.10+
- **Node.js**: 18+
- **MongoDB**: 6.0+

---

## Installation Time Estimates

| Task | Time (Fast Connection) | Time (Slow Connection) |
|------|----------------------|----------------------|
| Python packages | 1-2 minutes | 3-5 minutes |
| Node.js packages | 2-3 minutes | 5-10 minutes |
| Model training | 30-60 seconds | 30-60 seconds |
| **Total** | **4-6 minutes** | **9-16 minutes** |

---

## Disk Space Usage

| Component | Size |
|-----------|------|
| Backend (venv) | ~150 MB |
| Frontend (node_modules) | ~200 MB |
| MongoDB (database) | ~100 MB |
| Model files | ~10 MB |
| Source code | ~5 MB |
| **Total** | **~465 MB** |

---

## Network Requirements

### Development (localhost)
- No internet required after installation
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- MongoDB: mongodb://localhost:27017

### Production (deployment)
- Backend: Requires public IP/domain
- Frontend: CDN/hosting service
- MongoDB: Cloud database (MongoDB Atlas)
- HTTPS recommended for production

---

## Optional Enhancements

### Additional Python Packages
```
matplotlib==3.7.1                # Visualization
seaborn==0.12.2                  # Statistical plots
jupyter==1.0.0                   # Notebook interface
tensorboard==2.14.0              # Model monitoring
```

### Additional Node Packages
```
@tanstack/react-query            # Data fetching
react-router-dom                 # Routing
recharts                         # Charts
tailwindcss                      # CSS framework
```

---

## License Information

All dependencies are open-source with permissive licenses:
- MIT License: React, FastAPI, most packages
- BSD License: NumPy, scikit-learn, pandas
- Apache 2.0: MongoDB drivers

Safe for commercial use ✓

---

## Version Compatibility

| Software | Minimum | Recommended | Latest Tested |
|----------|---------|-------------|---------------|
| Python | 3.8 | 3.10 | 3.11 |
| Node.js | 16.0 | 18.0 | 20.0 |
| MongoDB | 5.0 | 6.0 | 7.0 |
| pip | 21.0 | 23.0 | 24.0 |
| npm | 8.0 | 9.0 | 10.0 |

---

## Browser Compatibility

### Supported Browsers
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Opera 76+

### Mobile Browsers
- ✅ iOS Safari 14+
- ✅ Chrome Mobile 90+
- ✅ Samsung Internet 14+

---

## Update Instructions

### Update Backend
```powershell
cd backend
.\venv\Scripts\activate
pip install --upgrade -r requirements.txt
```

### Update Frontend
```powershell
cd frontend
npm update
```

### Check for Outdated Packages
```powershell
# Python
pip list --outdated

# Node.js
npm outdated
```

---

**Last Updated**: March 2026
