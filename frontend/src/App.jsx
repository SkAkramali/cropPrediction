import { useState } from 'react'
import CropPredictor from './components/CropPredictor'
import './App.css'

function App() {
  return (
    <div className="App">
      <header className="app-header">
        <h1>🌾 Crop Recommendation System</h1>
        <p>AI-Powered Soil Analysis for Optimal Crop Selection</p>
      </header>
      <main>
        <CropPredictor />
      </main>
      <footer className="app-footer">
        <p>Powered by Machine Learning | Built with React & FastAPI</p>
      </footer>
    </div>
  )
}

export default App
