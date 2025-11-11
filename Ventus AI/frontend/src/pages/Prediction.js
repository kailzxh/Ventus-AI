// frontend/src/pages/Prediction.js
import React, { useState } from 'react';
import { useAQI } from '../context/AQIContext';
import PredictionForm from '../components/PredictionForm';
import PredictionResults from '../components/PredictionResults';
import FuturePredictions from '../components/FuturePredictions';
import LoadingSpinner from '../components/LoadingSpinner';
import ModelComparison from '../components/ModelComparison';
import { AlertCircle, TrendingUp } from 'lucide-react';

const Predictions = () => {
  const { selectedCity, predictAQI, getFuturePredictions, loading, apiError, systemStatus } = useAQI();
  const [predictionResult, setPredictionResult] = useState(null);
  const [futurePredictions, setFuturePredictions] = useState([]);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState(null);
  const [futureError, setFutureError] = useState(null);
  const [predictionFormData, setPredictionFormData] = useState(null);

  const handlePrediction = async (predictionData) => {
    try {
      setPredictionLoading(true);
      setPredictionResult(null);
      setPredictionError(null);
      setPredictionFormData(predictionData); // Store for model comparison
      
      if (process.env.NODE_ENV === 'development') {
        console.log('🎯 Starting prediction with data:', predictionData);
      }
      
      const result = await predictAQI(
        predictionData.city,
        predictionData.date,
        predictionData.modelType
      );
      
      if (process.env.NODE_ENV === 'development') {
        console.log('✅ Prediction completed:', result);
        console.log('📊 Requested model:', predictionData.modelType);
        console.log('📊 Model used (from API):', result.model_used);
        console.log('📈 Predicted AQI:', result.predicted_aqi);
        console.log('📋 Category:', result.category);
      }
      
      // Ensure result has the requested model information for display
      // Even if backend uses a different model, we show what was requested
      if (result) {
        result.requested_model = predictionData.modelType;
        // If model_used is not provided or is 'simple', still show the requested model
        if (!result.model_used || result.model_used === 'simple') {
          result.display_model = predictionData.modelType;
        } else {
          result.display_model = result.model_used;
        }
      }
      
      setPredictionResult(result);
    } catch (error) {
      const errorMessage = error.message || 'Prediction failed. Please try again.';
      console.error('❌ Prediction error:', error);
      setPredictionError(errorMessage);
      setPredictionResult(null);
    } finally {
      setPredictionLoading(false);
    }
  };

  const handleFuturePredictions = async (city, days) => {
    try {
      setPredictionLoading(true);
      setFutureError(null);
      
      if (process.env.NODE_ENV === 'development') {
        console.log('📅 Getting future predictions for:', city, 'Days:', days);
      }
      
      const results = await getFuturePredictions(city, days);
      
      if (process.env.NODE_ENV === 'development') {
        console.log('✅ Future predictions completed:', results);
      }
      
      // Ensure results is an array
      if (Array.isArray(results)) {
        setFuturePredictions(results);
      } else if (results && Array.isArray(results.predictions)) {
        setFuturePredictions(results.predictions);
      } else {
        setFuturePredictions([]);
      }
    } catch (error) {
      const errorMessage = error.message || 'Failed to get future predictions. Please try again.';
      console.error('❌ Future predictions error:', error);
      setFutureError(errorMessage);
      setFuturePredictions([]);
    } finally {
      setPredictionLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center">
            <TrendingUp className="mr-2" size={28} />
            AQI Predictions
          </h1>
          <p className="text-gray-600 mt-1">AI-powered air quality forecasting</p>
        </div>
      </div>

      {apiError && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 flex items-start">
          <AlertCircle size={20} className="text-red-400 mr-3 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h3 className="text-sm font-medium text-red-800">System Error</h3>
            <p className="text-sm text-red-700 mt-1">{apiError}</p>
          </div>
        </div>
      )}

      {!systemStatus?.initialized && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 flex items-start">
          <AlertCircle size={20} className="text-yellow-600 mr-3 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h3 className="text-sm font-medium text-yellow-800">System Initializing</h3>
            <p className="text-sm text-yellow-700 mt-1">
              The system is still loading. Predictions may be limited until initialization is complete.
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <PredictionForm 
            onSubmit={handlePrediction}
            loading={predictionLoading}
          />
          
          {predictionError && (
            <div className="bg-red-50 border border-red-200 rounded-md p-4 flex items-start">
              <AlertCircle size={20} className="text-red-400 mr-3 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="text-sm font-medium text-red-800">Prediction Error</h3>
                <p className="text-sm text-red-700 mt-1">{predictionError}</p>
              </div>
            </div>
          )}
          
          {predictionLoading && !predictionResult && (
            <div className="bg-white rounded-lg shadow p-6">
              <LoadingSpinner message="Generating prediction..." />
            </div>
          )}
          
          {predictionResult && (
            <PredictionResults result={predictionResult} />
          )}
        </div>

        <div className="space-y-6">
          <FuturePredictions 
            onPredict={handleFuturePredictions}
            predictions={futurePredictions}
            loading={predictionLoading}
            city={selectedCity}
          />
          
          {futureError && (
            <div className="bg-red-50 border border-red-200 rounded-md p-4 flex items-start">
              <AlertCircle size={20} className="text-red-400 mr-3 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="text-sm font-medium text-red-800">Forecast Error</h3>
                <p className="text-sm text-red-700 mt-1">{futureError}</p>
              </div>
            </div>
          )}

          {predictionFormData && predictionFormData.city && predictionFormData.date && (
            <ModelComparison 
              city={predictionFormData.city}
              date={predictionFormData.date}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default Predictions;