// frontend/src/components/PredictionCard.js
import React, { useState, useEffect } from 'react';
import { useAQI } from '../context/AQIContext';
import { Calendar, MapPin, Activity, AlertCircle, TrendingUp } from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';

const PredictionCard = () => {
  const { predictAQI, getAvailableModels, loading, apiError } = useAQI();
  const [prediction, setPrediction] = useState(null);
  const [city, setCity] = useState('bengaluru');
  const [date, setDate] = useState('');
  const [modelType, setModelType] = useState('auto');
  const [availableModels, setAvailableModels] = useState([]);
  const [predictionLoading, setPredictionLoading] = useState(false);

  useEffect(() => {
    // Set default date to tomorrow
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    setDate(tomorrow.toISOString().split('T')[0]);

    // Load available models
    loadAvailableModels();
  }, []);

  const loadAvailableModels = async () => {
    try {
      const models = await getAvailableModels();
      setAvailableModels(models.available_models || []);
      console.log('📊 Available models:', models);
    } catch (error) {
      console.error('Error loading models:', error);
      setAvailableModels(['auto', 'simple']);
    }
  };

  const handlePredict = async () => {
    if (!city || !date) {
      alert('Please select a city and date');
      return;
    }

    setPredictionLoading(true);
    try {
      console.log(`🔮 Predicting AQI for ${city} on ${date} using ${modelType}`);
      const result = await predictAQI(city, date, modelType);
      console.log('✅ Prediction result:', result);
      setPrediction(result);
    } catch (error) {
      console.error('❌ Prediction error:', error);
      setPrediction({
        error: error.message || 'Prediction failed',
        city,
        date,
        model_used: modelType
      });
    } finally {
      setPredictionLoading(false);
    }
  };

  const getModelDescription = (model) => {
    const descriptions = {
      'auto': 'Auto-select best available model',
      'nf_vae': 'Normalizing Flow VAE (Most Accurate)',
      'random_forest': 'Random Forest Ensemble',
      'gradient_boosting': 'Gradient Boosting Regressor',
      'simple': 'Simple Pattern-based Model'
    };
    return descriptions[model] || model;
  };

  const getHealthAdvice = (category) => {
    const advice = {
      'Good': 'Air quality is satisfactory, and air pollution poses little or no risk.',
      'Satisfactory': 'Air quality is acceptable. However, there may be a risk for some people.',
      'Moderate': 'Members of sensitive groups may experience health effects.',
      'Poor': 'Some members of the general public may experience health effects.',
      'Very Poor': 'Health alert: The risk of health effects is increased for everyone.',
      'Severe': 'Health warning of emergency conditions.'
    };
    return advice[category] || 'No health advice available.';
  };

  const getAQIColor = (aqi) => {
    if (aqi <= 50) return 'text-green-600 bg-green-50 border-green-200';
    if (aqi <= 100) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    if (aqi <= 200) return 'text-orange-600 bg-orange-50 border-orange-200';
    if (aqi <= 300) return 'text-red-600 bg-red-50 border-red-200';
    if (aqi <= 400) return 'text-purple-600 bg-purple-50 border-purple-200';
    return 'text-red-800 bg-red-100 border-red-300';
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-2 flex items-center">
        <Activity className="mr-2" size={24} />
        AQI Prediction
      </h2>
      <p className="text-gray-600 mb-6">AI-powered air quality forecasting</p>

      {/* Prediction Form */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <MapPin size={16} className="inline mr-1" />
            City
          </label>
          <select
            value={city}
            onChange={(e) => setCity(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          >
            <option value="delhi">Delhi</option>
            <option value="mumbai">Mumbai</option>
            <option value="bengaluru">Bengaluru</option>
            <option value="chennai">Chennai</option>
            <option value="kolkata">Kolkata</option>
            <option value="hyderabad">Hyderabad</option>
            <option value="pune">Pune</option>
            <option value="ahmedabad">Ahmedabad</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">{availableModels.length} models loaded</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <Calendar size={16} className="inline mr-1" />
            Prediction Date
          </label>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            min={new Date().toISOString().split('T')[0]}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <TrendingUp size={16} className="inline mr-1" />
            Prediction Model
          </label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          >
            {availableModels.map(model => (
              <option key={model} value={model}>
                {getModelDescription(model)}
              </option>
            ))}
          </select>
        </div>
      </div>

      <button
        onClick={handlePredict}
        disabled={predictionLoading}
        className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
      >
        {predictionLoading ? (
          <LoadingSpinner size="sm" message="Predicting..." />
        ) : (
          'Predict AQI'
        )}
      </button>

      {/* Prediction Tips */}
      <div className="mt-4 p-4 bg-blue-50 rounded-lg">
        <h3 className="font-semibold text-blue-900 mb-2">💡 Prediction Tips</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Auto model selection chooses the best available model</li>
          <li>• NF-VAE model provides the most accurate predictions</li>
          <li>• Predictions are based on historical patterns and real-time data</li>
          <li>• Accuracy decreases for predictions beyond 7 days</li>
        </ul>
      </div>

      {/* Prediction Results */}
      {prediction && (
        <div className="mt-6 border-t pt-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Results</h3>
          
          {prediction.error ? (
            <div className="bg-red-50 border border-red-200 rounded-md p-4">
              <div className="flex items-center">
                <AlertCircle size={20} className="text-red-400 mr-2" />
                <span className="text-red-800 font-medium">Prediction Error</span>
              </div>
              <p className="text-red-700 mt-2">{prediction.error}</p>
            </div>
          ) : (
            <>
              {/* Main Prediction Card */}
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 mb-4">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h4 className="text-xl font-bold text-gray-900">{prediction.city || city}</h4>
                    <p className="text-gray-600">{prediction.date || date}</p>
                  </div>
                  <div className="text-right">
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                      prediction.model_used?.includes('fallback') 
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-green-100 text-green-800'
                    }`}>
                      Model: {prediction.model_used?.toUpperCase()}
                    </span>
                    <p className="text-xs text-gray-500 mt-1">
                      {new Date().toLocaleDateString()}, {new Date().toLocaleTimeString()}
                    </p>
                  </div>
                </div>

                <div className="text-center mb-4">
                  <div className={`text-5xl font-bold mb-2 ${
                    getAQIColor(prediction.predicted_aqi).split(' ')[0]
                  }`}>
                    {Math.round(prediction.predicted_aqi)}
                  </div>
                  <div className="text-lg font-semibold text-gray-700">
                    Predicted AQI
                  </div>
                </div>

                <div className="text-center">
                  <span className={`inline-flex items-center px-4 py-2 rounded-full text-lg font-semibold ${
                    getAQIColor(prediction.predicted_aqi)
                  }`}>
                    {prediction.category || 'Unknown'}
                  </span>
                  <p className="text-sm text-gray-600 mt-2">
                    Air Quality Category
                  </p>
                </div>
              </div>

              {/* Health Advice */}
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
                <h5 className="font-semibold text-yellow-900 mb-2">Health Advice</h5>
                <p className="text-yellow-800 text-sm">
                  {getHealthAdvice(prediction.category)}
                </p>
              </div>

              {/* Prediction Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div className="bg-gray-50 rounded-lg p-3">
                  <span className="font-medium text-gray-700">Model Requested:</span>
                  <span className="ml-2 text-gray-900">{modelType.toUpperCase()}</span>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <span className="font-medium text-gray-700">Confidence:</span>
                  <span className="ml-2 text-green-600 font-medium">
                    {prediction.confidence || 'High (Based on historical patterns)'}
                  </span>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <span className="font-medium text-gray-700">Data Source:</span>
                  <span className="ml-2 text-gray-900">
                    {prediction.source || 'Historical + Real-time data'}
                  </span>
                </div>
                {prediction.model_loaded !== undefined && (
                  <div className="bg-gray-50 rounded-lg p-3">
                    <span className="font-medium text-gray-700">Model Loaded:</span>
                    <span className={`ml-2 font-medium ${
                      prediction.model_loaded ? 'text-green-600' : 'text-yellow-600'
                    }`}>
                      {prediction.model_loaded ? 'Yes' : 'No (Using fallback)'}
                    </span>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {/* API Error */}
      {apiError && (
        <div className="mt-4 bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex items-center">
            <AlertCircle size={20} className="text-red-400 mr-2" />
            <span className="text-red-800 font-medium">API Error</span>
          </div>
          <p className="text-red-700 mt-1">{apiError}</p>
        </div>
      )}
    </div>
  );
};

export default PredictionCard;