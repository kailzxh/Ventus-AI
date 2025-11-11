// frontend/src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AQIProvider } from './context/AQIContext';
import ErrorBoundary from './components/ErrorBoundary';
import Navbar from './components/Navbar';
import Dashboard from './pages/DashBoard.js';
import Predictions from './pages/Prediction.js';
import CityComparison from './pages/CityCamparsion.js';
import Analytics from './pages/Analytics.js';
import './App.css';

function App() {
  return (
    <ErrorBoundary>
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
    </ErrorBoundary>
  );
}

export default App;