// frontend/src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AQIProvider } from './context/AQIContext';
import Navbar from './components/Navbar';
import Dashboard from './pages/DashBoard.js';
import Predictions from './pages/Predictions';
import CityComparison from './pages/CityComparison';
import Analytics from './pages/AnalyticsClean';
import './App.css';

function App() {
  return (
    <AQIProvider>
      <Router>
        <div className="App min-h-screen bg-gray-50">
          <Navbar />
          <main className="container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/comparison" element={<CityComparison />} />
              <Route path="/analytics" element={<Analytics />} />
            </Routes>
          </main>
        </div>
      </Router>
    </AQIProvider>
  );
}

export default App;