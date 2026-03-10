import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const UploadCSVTab = () => {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState('');

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected && selected.name.endsWith('.csv')) {
      setFile(selected);
      setFileName(selected.name);
      setError(null);
    } else {
      setError('Please select a valid CSV file.');
      setFile(null);
      setFileName('');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a CSV file first.');
      return;
    }

    setLoading(true);
    setError(null);
    setPredictions([]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/predict-batch`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPredictions(response.data.predictions || []);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process CSV file.');
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = () => {
    if (predictions.length === 0) return;

    const headers = ['nitrogen', 'phosphorus', 'potassium', 'crop', 'confidence'];
    const csvRows = [headers.join(',')];
    predictions.forEach((p) => {
      csvRows.push(
        [p.nitrogen, p.phosphorus, p.potassium, p.crop, p.confidence].join(',')
      );
    });
    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'crop_predictions.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleReset = () => {
    setFile(null);
    setFileName('');
    setPredictions([]);
    setError(null);
    const input = document.getElementById('csv-input');
    if (input) input.value = '';
  };

  return (
    <div className="tab-content">
      <div className="card">
        <div className="card-header">
          <h2>📁 Bulk Prediction (CSV Upload)</h2>
          <p className="subtitle">Upload a CSV file with soil nutrient data for batch predictions</p>
        </div>

        <div className="upload-area">
          <div className="upload-box">
            <input
              type="file"
              id="csv-input"
              accept=".csv"
              onChange={handleFileChange}
              className="file-input"
            />
            <label htmlFor="csv-input" className="file-label">
              <span className="upload-icon">📤</span>
              <span>{fileName || 'Choose CSV file or drag & drop'}</span>
              <span className="upload-hint">Accepted format: .csv</span>
            </label>
          </div>

          <div className="csv-format-hint">
            <h4>Expected CSV format:</h4>
            <pre>nitrogen,phosphorus,potassium{'\n'}90,40,43{'\n'}70,35,50</pre>
          </div>
        </div>

        {error && <div className="error-msg">⚠️ {error}</div>}

        <div className="btn-group">
          <button
            className="btn btn-primary"
            onClick={handleUpload}
            disabled={loading || !file}
          >
            {loading ? <><span className="spinner" /> Processing...</> : <>🚀 Upload & Predict</>}
          </button>
          <button className="btn btn-secondary" onClick={handleReset} disabled={loading}>
            Clear
          </button>
          {predictions.length > 0 && (
            <button className="btn btn-success" onClick={downloadCSV}>
              ⬇️ Download Results CSV
            </button>
          )}
        </div>
      </div>

      {/* Results Table */}
      {predictions.length > 0 && (
        <div className="card animate-in">
          <div className="card-header">
            <h2>📊 Batch Results ({predictions.length} predictions)</h2>
          </div>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Nitrogen</th>
                  <th>Phosphorus</th>
                  <th>Potassium</th>
                  <th>Predicted Crop</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((p, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{p.nitrogen}</td>
                    <td>{p.phosphorus}</td>
                    <td>{p.potassium}</td>
                    <td className="crop-cell">🌾 {p.crop}</td>
                    <td>
                      <span className={`confidence-badge ${p.confidence >= 0.8 ? 'high' : p.confidence >= 0.6 ? 'medium' : 'low'}`}>
                        {(p.confidence * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadCSVTab;
