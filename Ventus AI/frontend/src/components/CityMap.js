// frontend/src/components/CityMap.js
import React from 'react';
import { MapPin } from 'lucide-react';
import EmptyState from './EmptyState';

const CityMap = ({ cities = [] }) => {
  if (!Array.isArray(cities) || cities.length === 0) {
    return (
      <EmptyState
        icon={MapPin}
        title="No city data available"
        message="City map will appear here once city data is loaded."
      />
    );
  }

  const validCities = cities
    .filter(c => c && (c.city || c.name))
    .slice(0, 10);

  if (validCities.length === 0) {
    return (
      <EmptyState
        icon={MapPin}
        title="No valid cities"
        message="Unable to display cities. Please ensure city data is available."
      />
    );
  }

  return (
    <div className="p-4">
      <div className="space-y-2">
        {validCities.map((city, index) => {
          const cityName = city.city || city.name || `City ${index + 1}`;
          const aqi = typeof city.aqi === 'number' ? Math.round(city.aqi) : '—';
          const category = city.category || 'Unknown';
          
          const getCategoryColor = (cat) => {
            const colors = {
              'Good': 'text-green-600',
              'Satisfactory': 'text-yellow-600',
              'Moderate': 'text-orange-600',
              'Poor': 'text-red-600',
              'Very Poor': 'text-purple-600',
              'Severe': 'text-red-700'
            };
            return colors[cat] || 'text-gray-600';
          };

          return (
            <div 
              key={index} 
              className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div className="flex items-center space-x-3">
                <MapPin size={18} className="text-gray-400" />
                <span className="font-medium text-gray-800">{cityName}</span>
              </div>
              <div className="flex items-center space-x-3">
                <span className={`font-semibold ${getCategoryColor(category)}`}>
                  AQI: {aqi}
                </span>
                <span className="text-xs text-gray-500 capitalize px-2 py-1 bg-white rounded">
                  {category}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      {cities.length > 10 && (
        <p className="text-xs text-gray-500 mt-4 text-center">
          Showing 10 of {cities.length} cities
        </p>
      )}
    </div>
  );
};

export default CityMap;
