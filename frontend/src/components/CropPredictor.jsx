/**
 * CropPredictor Component
 * ======================
 * Main component for crop prediction interface
 * 
 * Features:
 * - Input form for soil nutrients (N, P, K)
 * - Real-time validation
 * - API integration for predictions
 * - Results display with confidence score
 * - Prediction history
 * 
 * Author: AI Assistant
 * Date: March 2026
 */

import { useState, useEffect } from 'react';
import axios from 'axios';
import './CropPredictor.css';

// API Base URL
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const CropPredictor = () => {
  // =========================================================================
  // State Management
  // =========================================================================
  
  const [formData, setFormData] = useState({
    nitrogen: '',
    phosphorus: '',
    potassium: ''
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  
  // =========================================================================
  // Lifecycle - Fetch Model Info on Mount
  // =========================================================================
  
  useEffect(() => {
    fetchModelInfo();
    fetchHistory();
  }, []);
  
  // =========================================================================
  // API Functions
  // =========================================================================
  
  /**
   * Fetch model information from API
   */
  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API_URL}/model-info`);
      setModelInfo(response.data);
    } catch (err) {
      console.error('Error fetching model info:', err);
    }
  };
  
  /**
   * Fetch prediction history from API
   */
  const fetchHistory = async () => {
    try {
      const response = await axios.get(`${API_URL}/history?limit=10`);
      if (response.data.success) {
        setHistory(response.data.data);
      }
    } catch (err) {
      console.error('Error fetching history:', err);
    }
  };
  
  /**
   * Submit prediction request to API
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Reset states
    setError(null);
    setPrediction(null);
    setLoading(true);
    
    // Validate inputs
    const nitrogen = parseFloat(formData.nitrogen);
    const phosphorus = parseFloat(formData.phosphorus);
    const potassium = parseFloat(formData.potassium);
    
    if (isNaN(nitrogen) || isNaN(phosphorus) || isNaN(potassium)) {
      setError('Please enter valid numeric values for all fields');
      setLoading(false);
      return;
    }
    
    if (nitrogen < 0 || phosphorus < 0 || potassium < 0) {
      setError('Values must be positive numbers');
      setLoading(false);
      return;
    }
    
    if (nitrogen > 200 || phosphorus > 200 || potassium > 200) {
      setError('Values must be between 0 and 200');
      setLoading(false);
      return;
    }
    
    try {
      // Make API request
      const response = await axios.post(`${API_URL}/predict`, {
        nitrogen,
        phosphorus,
        potassium
      });
      
      // Set prediction result
      setPrediction(response.data);
      
      // Refresh history
      fetchHistory();
      
    } catch (err) {
      console.error('Prediction error:', err);
      
      if (err.response) {
        setError(err.response.data.detail || 'Prediction failed');
      } else if (err.request) {
        setError('Cannot connect to server. Please ensure the API is running.');
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setLoading(false);
    }
  };
  
  /**
   * Handle input changes
   */
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  /**
   * Reset form
   */
  const handleReset = () => {
    setFormData({
      nitrogen: '',
      phosphorus: '',
      potassium: ''
    });
    setPrediction(null);
    setError(null);
  };
  
  /**
   * Get confidence level text and color
   */
  const getConfidenceLevel = (score) => {
    if (score >= 0.9) return { text: 'Excellent', color: '#22c55e' };
    if (score >= 0.8) return { text: 'Very Good', color: '#84cc16' };
    if (score >= 0.7) return { text: 'Good', color: '#eab308' };
    if (score >= 0.6) return { text: 'Fair', color: '#f97316' };
    return { text: 'Low', color: '#ef4444' };
  };
  
  // =========================================================================
  // Render
  // =========================================================================
  
  return (
    <div className="crop-predictor">
      <div className="container">
        
        {/* Model Info Card */}
        {modelInfo && (
          <div className="info-card">
            <h3>📊 Model Information</h3>
            <div className="info-grid">
              <div className="info-item">
                <span className="label">Algorithm:</span>
                <span className="value">{modelInfo.model_name}</span>
              </div>
              <div className="info-item">
                <span className="label">Accuracy:</span>
                <span className="value">{(modelInfo.accuracy * 100).toFixed(2)}%</span>
              </div>
              <div className="info-item">
                <span className="label">Crops Supported:</span>
                <span className="value">{modelInfo.classes.length}</span>
              </div>
            </div>
          </div>
        )}
        
        {/* Input Form */}
        <div className="form-card">
          <h2>Enter Soil Nutrient Values</h2>
          <p className="subtitle">Provide the NPK values from your soil test</p>
          
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="nitrogen">
                <span className="nutrient-symbol">N</span>
                Nitrogen (N)
              </label>
              <input
                type="number"
                id="nitrogen"
                name="nitrogen"
                value={formData.nitrogen}
                onChange={handleInputChange}
                placeholder="Enter nitrogen value (0-200)"
                min="0"
                max="200"
                step="0.01"
                required
              />
              <span className="helper-text">Typical range: 0-200 mg/kg</span>
            </div>
            
            <div className="form-group">
              <label htmlFor="phosphorus">
                <span className="nutrient-symbol">P</span>
                Phosphorus (P)
              </label>
              <input
                type="number"
                id="phosphorus"
                name="phosphorus"
                value={formData.phosphorus}
                onChange={handleInputChange}
                placeholder="Enter phosphorus value (0-200)"
                min="0"
                max="200"
                step="0.01"
                required
              />
              <span className="helper-text">Typical range: 0-200 mg/kg</span>
            </div>
            
            <div className="form-group">
              <label htmlFor="potassium">
                <span className="nutrient-symbol">K</span>
                Potassium (K)
              </label>
              <input
                type="number"
                id="potassium"
                name="potassium"
                value={formData.potassium}
                onChange={handleInputChange}
                placeholder="Enter potassium value (0-200)"
                min="0"
                max="200"
                step="0.01"
                required
              />
              <span className="helper-text">Typical range: 0-200 mg/kg</span>
            </div>
            
            {/* Error Message */}
            {error && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                {error}
              </div>
            )}
            
            {/* Action Buttons */}
            <div className="button-group">
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <span>🔍</span>
                    Predict Crop
                  </>
                )}
              </button>
              
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={loading}
              >
                Reset
              </button>
            </div>
          </form>
        </div>
        
        {/* Prediction Result */}
        {prediction && (
          <div className="result-card">
            <h2>🎯 Prediction Result</h2>
            
            <div className="result-main">
              <div className="crop-recommendation">
                <span className="label">Recommended Crop:</span>
                <span className="crop-name">
                  {prediction.prediction.recommended_crop.toUpperCase()}
                </span>
              </div>
              
              <div className="confidence-section">
                <div className="confidence-header">
                  <span className="label">Confidence Score:</span>
                  <span 
                    className="confidence-score"
                    style={{ 
                      color: getConfidenceLevel(prediction.prediction.confidence_score).color 
                    }}
                  >
                    {(prediction.prediction.confidence_score * 100).toFixed(2)}%
                  </span>
                </div>
                
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill"
                    style={{ 
                      width: `${prediction.prediction.confidence_score * 100}%`,
                      backgroundColor: getConfidenceLevel(prediction.prediction.confidence_score).color
                    }}
                  ></div>
                </div>
                
                <div className="confidence-label">
                  {getConfidenceLevel(prediction.prediction.confidence_score).text} Confidence
                </div>
              </div>
            </div>
            
            <div className="result-details">
              <h3>Input Values:</h3>
              <div className="values-grid">
                <div className="value-item">
                  <span className="nutrient">N</span>
                  <span className="amount">{prediction.input.nitrogen}</span>
                </div>
                <div className="value-item">
                  <span className="nutrient">P</span>
                  <span className="amount">{prediction.input.phosphorus}</span>
                </div>
                <div className="value-item">
                  <span className="nutrient">K</span>
                  <span className="amount">{prediction.input.potassium}</span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* History Section */}
        <div className="history-section">
          <div className="history-header">
            <h3>📜 Recent Predictions</h3>
            <button 
              className="btn-toggle"
              onClick={() => setShowHistory(!showHistory)}
            >
              {showHistory ? 'Hide' : 'Show'} History
            </button>
          </div>
          
          {showHistory && (
            <div className="history-list">
              {history.length === 0 ? (
                <p className="no-history">No predictions yet</p>
              ) : (
                history.map((item, index) => (
                  <div key={index} className="history-item">
                    <div className="history-crop">{item.recommended_crop}</div>
                    <div className="history-values">
                      N: {item.nitrogen}, P: {item.phosphorus}, K: {item.potassium}
                    </div>
                    <div className="history-confidence">
                      {(item.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="history-date">
                      {new Date(item.timestamp).toLocaleString()}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
        
      </div>
    </div>
  );
};

export default CropPredictor;
