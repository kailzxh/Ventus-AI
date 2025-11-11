// frontend/src/components/PredictionForm.js
import React, { useState, useEffect } from 'react';
import { useAQI } from '../context/AQIContext';
import { Calendar, MapPin, Cpu, AlertCircle } from 'lucide-react';

const PredictionForm = ({ onSubmit, loading }) => {
  const { availableCities, systemStatus, availableModels } = useAQI();
  const [formData, setFormData] = useState({
    city: 'Delhi',
    date: new Date().toISOString().split('T')[0],
    modelType: 'auto'
  });

  // Update available models when they load
  useEffect(() => {
    if (Array.isArray(availableModels) && availableModels.length > 0) {
      // Ensure formData.modelType is valid
      if (!availableModels.includes(formData.modelType) && formData.modelType !== 'auto') {
        setFormData(prev => ({
          ...prev,
          modelType: availableModels.includes('nf_vae') ? 'nf_vae' : availableModels[0] || 'auto'
        }));
      }
    }
  }, [availableModels, formData.modelType]);

  // Ensure cities is always an array
  const cities = Array.isArray(availableCities) && availableCities.length > 0
    ? [...availableCities].filter(c => c && c !== 'Unknown').sort() 
    : ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'].sort();

  // Ensure availableModels is always an array
  const validModels = Array.isArray(availableModels) ? availableModels : [];
  
  const models = [
    { value: 'auto', label: 'Auto (Recommended)', description: 'Automatically select the best available model' },
    { value: 'nf_vae', label: 'NF-VAE (Advanced)', description: 'Most accurate - Gaussian Mixture Model' },
    { value: 'random_forest', label: 'Random Forest', description: 'Ensemble tree-based model' },
    { value: 'gradient_boosting', label: 'Gradient Boosting', description: 'Boosting ensemble model' },
    { value: 'simple', label: 'Simple Model', description: 'Basic pattern-based prediction' }
  ].filter(model => 
    model.value === 'auto' || 
    validModels.length === 0 || 
    validModels.includes(model.value)
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!systemStatus.initialized) {
      alert('System is not initialized. Please wait for the system to load.');
      return;
    }
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
      
      {!systemStatus.initialized && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md flex items-start">
          <AlertCircle size={16} className="text-yellow-600 mt-0.5 mr-2 flex-shrink-0" />
          <div>
            <p className="text-sm text-yellow-800">
              <strong>System Initializing:</strong> Predictions may be limited until system is fully loaded.
            </p>
          </div>
        </div>
      )}

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
            disabled={!systemStatus.initialized}
          >
            {cities.map(city => (
              <option key={city} value={city}>{city}</option>
            ))}
          </select>
          <p className="text-xs text-gray-500 mt-1">
            {cities.length} cities available
          </p>
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
            disabled={!systemStatus.initialized}
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
            disabled={!systemStatus.initialized}
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
          <p className="text-xs text-gray-400 mt-1">
            {availableModels.length} models loaded
          </p>
        </div>

        <button
          type="submit"
          disabled={loading || !systemStatus.initialized}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Predicting...
            </div>
          ) : !systemStatus.initialized ? (
            'System Loading...'
          ) : (
            'Predict AQI'
          )}
        </button>
      </form>

      <div className="mt-4 p-3 bg-blue-50 rounded-md">
        <h4 className="font-medium text-blue-800 mb-1">💡 Prediction Tips</h4>
        <ul className="text-xs text-blue-700 space-y-1">
          <li>• Auto model selection chooses the best available model</li>
          <li>• NF-VAE model provides the most accurate predictions</li>
          <li>• Predictions are based on historical patterns and real-time data</li>
          <li>• Accuracy decreases for predictions beyond 7 days</li>
          {!systemStatus.initialized && (
            <li className="text-yellow-700">• System is initializing... predictions may be limited</li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default PredictionForm;