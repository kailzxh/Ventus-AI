// frontend/src/components/AQICard.js
import React from 'react';
import { AlertCircle } from 'lucide-react';

const AQICard = ({ city }) => {
  // Validate city data
  if (!city || (!city.city && !city.name)) {
    return (
      <div className="bg-white rounded-lg shadow-md p-4 border-l-4 border-gray-300">
        <div className="flex items-center text-gray-500">
          <AlertCircle size={20} className="mr-2" />
          <span className="text-sm">Invalid city data</span>
        </div>
      </div>
    );
  }

  const cityName = city.city || city.name || 'Unknown';
  const aqi = typeof city.aqi === 'number' && !isNaN(city.aqi) ? city.aqi : null;
  const category = city.category || 'Unknown';
  const pm25 = city.pm25 !== undefined && city.pm25 !== null ? city.pm25 : null;
  const pm10 = city.pm10 !== undefined && city.pm10 !== null ? city.pm10 : null;
  const timestamp = city.timestamp ? new Date(city.timestamp) : new Date();

  const getAQIColor = (aqiValue) => {
    if (!aqiValue) return 'bg-gray-400';
    if (aqiValue <= 50) return 'bg-green-500';
    if (aqiValue <= 100) return 'bg-yellow-500';
    if (aqiValue <= 200) return 'bg-orange-500';
    if (aqiValue <= 300) return 'bg-red-500';
    if (aqiValue <= 400) return 'bg-purple-500';
    return 'bg-red-700';
  };

  const getAQITextColor = (aqiValue) => {
    if (!aqiValue) return 'text-gray-500';
    if (aqiValue <= 50) return 'text-green-600';
    if (aqiValue <= 100) return 'text-yellow-600';
    if (aqiValue <= 200) return 'text-orange-600';
    if (aqiValue <= 300) return 'text-red-600';
    if (aqiValue <= 400) return 'text-purple-600';
    return 'text-red-700';
  };

  const getBorderColor = (aqiValue) => {
    if (!aqiValue) return 'border-gray-400';
    if (aqiValue <= 50) return 'border-green-500';
    if (aqiValue <= 100) return 'border-yellow-500';
    if (aqiValue <= 200) return 'border-orange-500';
    if (aqiValue <= 300) return 'border-red-500';
    if (aqiValue <= 400) return 'border-purple-500';
    return 'border-red-700';
  };

  const getAQIIcon = (cat) => {
    const categoryMap = {
      'Good': '😊',
      'Satisfactory': '🙂',
      'Moderate': '😐',
      'Poor': '😷',
      'Very Poor': '😨',
      'Severe': '💀'
    };
    return categoryMap[cat] || '❓';
  };

  const progressWidth = aqi ? Math.min((aqi / 500) * 100, 100) : 0;

  return (
    <div className={`bg-white rounded-lg shadow-md p-4 border-l-4 ${getBorderColor(aqi)} hover:shadow-lg transition-all duration-200`}>
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-semibold text-gray-800 text-lg truncate flex-1">{cityName}</h3>
        <span className="text-2xl ml-2 flex-shrink-0">{getAQIIcon(category)}</span>
      </div>
      
      <div className="flex items-center justify-between mb-3">
        <div className="flex-1">
          <div className={`text-3xl font-bold ${getAQITextColor(aqi)}`}>
            {aqi !== null ? Math.round(aqi) : '—'}
          </div>
          <div className="text-sm text-gray-600 capitalize mt-1">{category}</div>
        </div>
        
        <div className="text-right text-sm text-gray-600 ml-4">
          {pm25 !== null && <div>PM2.5: {typeof pm25 === 'number' ? Math.round(pm25) : pm25}</div>}
          {pm10 !== null && <div>PM10: {typeof pm10 === 'number' ? Math.round(pm10) : pm10}</div>}
          {pm25 === null && pm10 === null && <div className="text-gray-400">No PM data</div>}
        </div>
      </div>
      
      {aqi !== null && (
        <div className="mt-3 w-full bg-gray-200 rounded-full h-2">
          <div 
            className={`h-2 rounded-full transition-all duration-300 ${getAQIColor(aqi)}`}
            style={{ width: `${progressWidth}%` }}
          ></div>
        </div>
      )}
      
      <div className="mt-2 text-xs text-gray-500">
        Updated: {timestamp.toLocaleTimeString()}
      </div>
    </div>
  );
};

export default AQICard; 