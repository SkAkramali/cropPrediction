import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const PredictTab = () => {
  const [formData, setFormData] = useState({ nitrogen: '', phosphorus: '', potassium: '' });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setPrediction(null);
    setLoading(true);

    const n = parseFloat(formData.nitrogen);
    const p = parseFloat(formData.phosphorus);
    const k = parseFloat(formData.potassium);

    if (isNaN(n) || isNaN(p) || isNaN(k)) {
      setError('Please enter valid numeric values for all fields');
      setLoading(false);
      return;
    }
    if (n < 0 || p < 0 || k < 0 || n > 200 || p > 200 || k > 200) {
      setError('Values must be between 0 and 200');
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        nitrogen: n, phosphorus: p, potassium: k,
      });
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Cannot connect to server.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({ nitrogen: '', phosphorus: '', potassium: '' });
    setPrediction(null);
    setError(null);
  };

  const getConfidenceLevel = (score) => {
    if (score >= 0.9) return { text: 'Excellent', color: '#22c55e' };
    if (score >= 0.8) return { text: 'Very Good', color: '#84cc16' };
    if (score >= 0.7) return { text: 'Good', color: '#eab308' };
    if (score >= 0.6) return { text: 'Fair', color: '#f97316' };
    return { text: 'Low', color: '#ef4444' };
  };

  return (
    <div className="tab-content">
      {/* Prediction Form */}
      <div className="card">
        <div className="card-header">
          <h2>🌾 Single Crop Prediction</h2>
          <p className="subtitle">Enter soil nutrient values to get a crop recommendation</p>
        </div>
        <form onSubmit={handleSubmit}>
          {['nitrogen', 'phosphorus', 'potassium'].map((field) => {
            const labels = { nitrogen: 'Nitrogen (N)', phosphorus: 'Phosphorus (P)', potassium: 'Potassium (K)' };
            const symbols = { nitrogen: 'N', phosphorus: 'P', potassium: 'K' };
            return (
              <div className="form-group" key={field}>
                <label htmlFor={field}>
                  <span className="nutrient-badge">{symbols[field]}</span>
                  {labels[field]}
                </label>
                <input
                  type="number"
                  id={field}
                  name={field}
                  value={formData[field]}
                  onChange={handleInputChange}
                  placeholder={`Enter ${field} value (0-200)`}
                  min="0" max="200" step="0.01" required
                />
                <span className="helper-text">Typical range: 0-200 mg/kg</span>
              </div>
            );
          })}

          {error && <div className="error-msg">⚠️ {error}</div>}

          <div className="btn-group">
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? <><span className="spinner" /> Analyzing...</> : <>🔍 Predict Crop</>}
            </button>
            <button type="button" className="btn btn-secondary" onClick={handleReset} disabled={loading}>
              Reset
            </button>
          </div>
        </form>
      </div>

      {/* Result Card */}
      {prediction && (
        <div className="card result-card animate-in">
          <h2>🎯 Prediction Result</h2>
          <div className="result-banner">
            <span className="result-label">Recommended Crop</span>
            <span className="result-crop">
              🌾 {prediction.prediction.recommended_crop.toUpperCase()}
            </span>
          </div>

          <div className="confidence-section">
            <div className="confidence-row">
              <span>Confidence Score</span>
              <span
                className="confidence-value"
                style={{ color: getConfidenceLevel(prediction.prediction.confidence_score).color }}
              >
                {(prediction.prediction.confidence_score * 100).toFixed(1)}%
              </span>
            </div>
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{
                  width: `${prediction.prediction.confidence_score * 100}%`,
                  backgroundColor: getConfidenceLevel(prediction.prediction.confidence_score).color,
                }}
              />
            </div>
            <div className="confidence-label">
              {getConfidenceLevel(prediction.prediction.confidence_score).text} Confidence
            </div>
          </div>

          <div className="input-summary">
            <h3>Input Values</h3>
            <div className="npk-grid">
              {[
                { label: 'N', value: prediction.input.nitrogen },
                { label: 'P', value: prediction.input.phosphorus },
                { label: 'K', value: prediction.input.potassium },
              ].map((item) => (
                <div className="npk-item" key={item.label}>
                  <span className="npk-label">{item.label}</span>
                  <span className="npk-value">{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictTab;
