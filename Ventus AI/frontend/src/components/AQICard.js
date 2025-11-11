// frontend/src/components/AQICard.js
import React from 'react';

const AQICard = ({ city }) => {
  const getAQIColor = (aqi) => {
    if (aqi <= 50) return 'bg-green-500';
    if (aqi <= 100) return 'bg-yellow-500';
    if (aqi <= 200) return 'bg-orange-500';
    if (aqi <= 300) return 'bg-red-500';
    if (aqi <= 400) return 'bg-purple-500';
    return 'bg-maroon-500';
  };

  const getAQITextColor = (aqi) => {
    if (aqi <= 50) return 'text-green-500';
    if (aqi <= 100) return 'text-yellow-500';
    if (aqi <= 200) return 'text-orange-500';
    if (aqi <= 300) return 'text-red-500';
    if (aqi <= 400) return 'text-purple-500';
    return 'text-maroon-500';
  };

  const getAQIIcon = (category) => {
    switch (category) {
      case 'Good':
        return '😊';
      case 'Satisfactory':
        return '🙂';
      case 'Moderate':
        return '😐';
      case 'Poor':
        return '😷';
      case 'Very Poor':
        return '😨';
      case 'Severe':
        return '💀';
      default:
        return '❓';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-4 border-l-4 border-blue-500 hover:shadow-lg transition-shadow">
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-semibold text-gray-800 text-lg">{city.city}</h3>
        <span className="text-2xl">{getAQIIcon(city.category)}</span>
      </div>
      
      <div className="flex items-center justify-between">
        <div>
          <div className={`text-2xl font-bold ${getAQITextColor(city.aqi)}`}>
            {city.aqi}
          </div>
          <div className="text-sm text-gray-600 capitalize">{city.category}</div>
        </div>
        
        <div className="text-right text-sm text-gray-600">
          <div>PM2.5: {city.pm25 || 'N/A'}</div>
          <div>PM10: {city.pm10 || 'N/A'}</div>
        </div>
      </div>
      
      <div className="mt-3 w-full bg-gray-200 rounded-full h-2">
        <div 
          className={`h-2 rounded-full ${getAQIColor(city.aqi)}`}
          style={{ width: `${Math.min((city.aqi / 500) * 100, 100)}%` }}
        ></div>
      </div>
      
      <div className="mt-2 text-xs text-gray-500">
        Updated: {new Date(city.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
};

export default AQICard; 