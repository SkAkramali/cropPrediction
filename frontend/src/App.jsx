import { useState } from 'react'
import Navbar from './components/Navbar'
import PredictTab from './components/PredictTab'
import UploadCSVTab from './components/UploadCSVTab'
import HistoryTab from './components/HistoryTab'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('predict')

  const renderTab = () => {
    switch (activeTab) {
      case 'predict': return <PredictTab />
      case 'upload': return <UploadCSVTab />
      case 'history': return <HistoryTab />
      default: return <PredictTab />
    }
  }

  return (
    <div className="App">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        {renderTab()}
      </main>
      <footer className="app-footer">
        <p>Powered by Machine Learning | Built with React & FastAPI</p>
      </footer>
    </div>
  )
}

export default App
