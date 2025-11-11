// frontend/src/components/FuturePredictions.js
import React, { useState } from 'react';
import { useAQI } from '../context/AQIContext';
import { Calendar, TrendingUp } from 'lucide-react';

const FuturePredictions = ({ onPredict, predictions, loading, city }) => {
  const { selectedCity, availableCities } = useAQI();
  const [days, setDays] = useState(7);
  
  // Ensure availableCities is an array and filter out invalid values
  const validCities = Array.isArray(availableCities) 
    ? availableCities.filter(c => c && c !== 'Unknown')
    : ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'];
  
  // Set default city
  const defaultCity = city || selectedCity || validCities[0] || 'Delhi';
  const [selectedCityLocal, setSelectedCityLocal] = useState(defaultCity);
  
  // Update selectedCityLocal when city prop or availableCities change
  React.useEffect(() => {
    if (city && validCities.includes(city)) {
      setSelectedCityLocal(city);
    } else if (selectedCity && validCities.includes(selectedCity)) {
      setSelectedCityLocal(selectedCity);
    } else if (validCities.length > 0 && !validCities.includes(selectedCityLocal)) {
      setSelectedCityLocal(validCities[0]);
    }
  }, [city, selectedCity, validCities]);

  const handlePredict = () => {
    onPredict(selectedCityLocal, days);
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
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">City</label>
          <select
            value={selectedCityLocal}
            onChange={(e) => setSelectedCityLocal(e.target.value)}
            disabled={loading || validCities.length === 0}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {validCities.length > 0 ? (
              validCities.map(cityName => (
                <option key={cityName} value={cityName}>{cityName}</option>
              ))
            ) : (
              <option value="">No cities available</option>
            )}
          </select>
        </div>
        
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
              Forecast for {selectedCityLocal}
            </h3>
            
            {predictions.map((prediction, index) => {
              // Validate prediction data
              const date = prediction.date || prediction.Date || '';
              const aqi = typeof prediction.predicted_aqi === 'number' 
                ? Math.round(prediction.predicted_aqi) 
                : prediction.predicted_aqi || '—';
              const category = prediction.category || 'Unknown';
              
              // Parse date safely
              let formattedDate = 'N/A';
              try {
                if (date) {
                  formattedDate = new Date(date).toLocaleDateString('en-US', { 
                    weekday: 'short', 
                    month: 'short', 
                    day: 'numeric' 
                  });
                }
              } catch (e) {
                formattedDate = date || 'N/A';
              }

              return (
                <div key={prediction.date || index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                  <div className="flex items-center space-x-3 flex-1">
                    <div className="text-sm font-medium text-gray-900 min-w-[100px]">
                      {formattedDate}
                    </div>
                    <div className={`px-2 py-1 rounded-full text-xs font-medium whitespace-nowrap ${
                      category === 'Good' ? 'bg-green-100 text-green-800' :
                      category === 'Satisfactory' ? 'bg-yellow-100 text-yellow-800' :
                      category === 'Moderate' ? 'bg-orange-100 text-orange-800' :
                      category === 'Poor' ? 'bg-red-100 text-red-800' :
                      category === 'Very Poor' ? 'bg-purple-100 text-purple-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {category}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <span className="text-lg font-bold text-gray-900">
                      {typeof aqi === 'number' ? aqi : aqi}
                    </span>
                    {typeof aqi === 'number' && (
                      <span className="text-sm text-gray-500">
                        {getTrendIcon(
                          aqi,
                          index > 0 && typeof predictions[index - 1].predicted_aqi === 'number' 
                            ? predictions[index - 1].predicted_aqi 
                            : null
                        )}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {!loading && predictions.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <Calendar size={48} className="mx-auto mb-2 text-gray-300" />
            <p className="text-sm">Generate {days}-day forecast to see predictions</p>
            <p className="text-xs text-gray-400 mt-2">Select a city and click Generate to view forecast</p>
          </div>
        )}

        {loading && (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
            <p className="text-sm text-gray-500 mt-2">Loading forecast...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default FuturePredictions;