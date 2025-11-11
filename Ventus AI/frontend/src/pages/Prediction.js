// frontend/src/pages/Predictions.js
import React, { useState } from 'react';
import { useAQI } from '../context/AQIContext';
import PredictionForm from '../components/PredictionForm';
import PredictionResults from '../components/PredictionResults';
import FuturePredictions from '../components/FuturePredictions';

const Predictions = () => {
  const { selectedCity, predictAQI, getFuturePredictions, predictions } = useAQI();
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [futurePredictions, setFuturePredictions] = useState([]);

  const handlePrediction = async (predictionData) => {
    try {
      setLoading(true);
      const result = await predictAQI(
        predictionData.city,
        predictionData.date,
        predictionData.modelType
      );
      setPredictionResult(result);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFuturePredictions = async (city, days) => {
    try {
      setLoading(true);
      const results = await getFuturePredictions(city, days);
      setFuturePredictions(results);
    } catch (error) {
      console.error('Future predictions error:', error);
      alert('Failed to get future predictions.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">AQI Predictions</h1>
        <p className="text-gray-600">AI-powered air quality forecasting</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <PredictionForm 
            onSubmit={handlePrediction}
            loading={loading}
          />
          
          {predictionResult && (
            <PredictionResults result={predictionResult} />
          )}
        </div>

        <div>
          <FuturePredictions 
            onPredict={handleFuturePredictions}
            predictions={futurePredictions}
            loading={loading}
          />
        </div>
      </div>
    </div>
  );
};

export default Predictions;