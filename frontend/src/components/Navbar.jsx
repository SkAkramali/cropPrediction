import { useEffect, useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const Navbar = ({ activeTab, setActiveTab }) => {
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    axios.get(`${API_URL}/model-info`).then((res) => {
      setModelInfo(res.data);
    }).catch(() => {});
  }, []);

  const tabs = [
    { id: 'predict', label: '🌾 Predict', icon: '🔍' },
    { id: 'upload', label: '📁 Bulk CSV', icon: '📤' },
    { id: 'history', label: '📜 History', icon: '📊' },
  ];

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <span className="brand-icon">🌾</span>
          <div className="brand-text">
            <h1>Crop Recommendation</h1>
            <span className="brand-subtitle">AI-Powered Soil Analysis</span>
          </div>
        </div>

        <div className="navbar-tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {modelInfo && (
          <div className="navbar-info">
            <span className="model-badge">
              {modelInfo.model_name} • {(modelInfo.accuracy * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
