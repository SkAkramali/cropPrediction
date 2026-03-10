import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const HistoryTab = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/history?limit=100`);
      if (response.data.success) {
        setHistory(response.data.data);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch history.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  return (
    <div className="tab-content">
      <div className="card">
        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2>📜 Prediction History</h2>
            <p className="subtitle">Previous crop predictions stored in the database</p>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={fetchHistory} disabled={loading}>
            {loading ? 'Loading...' : '🔄 Refresh'}
          </button>
        </div>

        {error && <div className="error-msg">⚠️ {error}</div>}

        {!loading && history.length === 0 && !error && (
          <div className="empty-state">
            <span className="empty-icon">📭</span>
            <p>No predictions yet. Make a prediction to see it here!</p>
          </div>
        )}

        {history.length > 0 && (
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
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item, i) => (
                  <tr key={item._id || i}>
                    <td>{i + 1}</td>
                    <td>{item.nitrogen}</td>
                    <td>{item.phosphorus}</td>
                    <td>{item.potassium}</td>
                    <td className="crop-cell">🌾 {item.recommended_crop}</td>
                    <td>
                      <span className={`confidence-badge ${item.confidence >= 0.8 ? 'high' : item.confidence >= 0.6 ? 'medium' : 'low'}`}>
                        {(item.confidence * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="date-cell">
                      {new Date(item.timestamp).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="history-summary">
          <span>Total Records: <strong>{history.length}</strong></span>
        </div>
      </div>
    </div>
  );
};

export default HistoryTab;
