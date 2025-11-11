// frontend/src/components/PredictionForm.js
import React, { useState } from 'react';
import { useAQI } from '../context/AQIContext';
import { Calendar, MapPin, Cpu } from 'lucide-react';

const PredictionForm = ({ onSubmit, loading }) => {
  const { currentAQI } = useAQI();
  const [formData, setFormData] = useState({
    city: 'Delhi',
    date: new Date().toISOString().split('T')[0],
    modelType: 'nf_vae'
  });

  // Use fixed list of supported cities
  const cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'].sort();
  const models = [
    { value: 'nf_vae', label: 'NF-VAE (Recommended)', description: 'Most accurate - Gaussian Mixture Model' },
    { value: 'random_forest', label: 'Random Forest', description: 'Ensemble tree-based model' },
    { value: 'gradient_boosting', label: 'Gradient Boosting', description: 'Boosting ensemble model' },
    { value: 'xgboost', label: 'XGBoost', description: 'Optimized gradient boosting' }
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const minDate = new Date().toISOString().split('T')[0];
  const maxDate = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4 flex items-center">
        <Cpu className="mr-2" size={20} />
        AQI Prediction
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
            <MapPin size={16} className="mr-1" />
            City
          </label>
          <select
            value={formData.city}
            onChange={(e) => handleChange('city', e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            required
          >
            {cities.map(city => (
              <option key={city} value={city}>{city}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
            <Calendar size={16} className="mr-1" />
            Prediction Date
          </label>
          <input
            type="date"
            value={formData.date}
            onChange={(e) => handleChange('date', e.target.value)}
            min={minDate}
            max={maxDate}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Prediction Model
          </label>
          <select
            value={formData.modelType}
            onChange={(e) => handleChange('modelType', e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            {models.map(model => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
          <p className="text-xs text-gray-500 mt-1">
            {models.find(m => m.value === formData.modelType)?.description}
          </p>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Predicting...
            </div>
          ) : (
            'Predict AQI'
          )}
        </button>
      </form>

      <div className="mt-4 p-3 bg-blue-50 rounded-md">
        <h4 className="font-medium text-blue-800 mb-1">💡 Prediction Tips</h4>
        <ul className="text-xs text-blue-700 space-y-1">
          <li>• NF-VAE model provides the most accurate predictions</li>
          <li>• Predictions are based on historical patterns and real-time data</li>
          <li>• Accuracy decreases for predictions beyond 7 days</li>
        </ul>
      </div>
    </div>
  );
};

export default PredictionForm;