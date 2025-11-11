// frontend/src/components/FuturePredictions.js
import React, { useState } from 'react';
import { useAQI } from '../context/AQIContext';
import { Calendar, TrendingUp } from 'lucide-react';

const FuturePredictions = ({ onPredict, predictions, loading }) => {
  const { selectedCity } = useAQI();
  const [days, setDays] = useState(7);

  const handlePredict = () => {
    onPredict(selectedCity, days);
  };

  const getTrendIcon = (currentAqi, previousAqi) => {
    if (!previousAqi) return '→';
    if (currentAqi > previousAqi) return '↗️';
    if (currentAqi < previousAqi) return '↘️';
    return '→';
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4 flex items-center">
        <Calendar className="mr-2" size={20} />
        7-Day Forecast
      </h2>

      <div className="space-y-4">
        <div className="flex space-x-2">
          <select
            value={days}
            onChange={(e) => setDays(parseInt(e.target.value))}
            className="flex-1 p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          >
            <option value={3}>3 Days</option>
            <option value={7}>7 Days</option>
            <option value={14}>14 Days</option>
          </select>
          
          <button
            onClick={handlePredict}
            disabled={loading}
            className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50 transition-colors"
          >
            {loading ? '...' : 'Generate'}
          </button>
        </div>

        {predictions.length > 0 && (
          <div className="space-y-3">
            <h3 className="font-medium text-gray-900 flex items-center">
              <TrendingUp size={16} className="mr-1" />
              Forecast for {selectedCity}
            </h3>
            
            {predictions.map((prediction, index) => (
              <div key={prediction.date} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="text-sm font-medium text-gray-900">
                    {new Date(prediction.date).toLocaleDateString('en-US', { 
                      weekday: 'short', 
                      month: 'short', 
                      day: 'numeric' 
                    })}
                  </div>
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                    prediction.category === 'Good' ? 'bg-green-100 text-green-800' :
                    prediction.category === 'Satisfactory' ? 'bg-yellow-100 text-yellow-800' :
                    prediction.category === 'Moderate' ? 'bg-orange-100 text-orange-800' :
                    prediction.category === 'Poor' ? 'bg-red-100 text-red-800' :
                    'bg-purple-100 text-purple-800'
                  }`}>
                    {prediction.category}
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className="text-lg font-bold text-gray-900">
                    {prediction.predicted_aqi}
                  </span>
                  <span className="text-sm text-gray-500">
                    {getTrendIcon(
                      prediction.predicted_aqi,
                      index > 0 ? predictions[index - 1].predicted_aqi : null
                    )}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {predictions.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <Calendar size={48} className="mx-auto mb-2 text-gray-300" />
            <p>Generate {days}-day forecast to see predictions</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default FuturePredictions;