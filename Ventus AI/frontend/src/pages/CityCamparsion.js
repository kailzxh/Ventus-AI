// frontend/src/pages/CityComparison.js
import React, { useState, useMemo } from 'react';
import { useAQI } from '../context/AQIContext';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const CityComparison = () => {
  const { currentAQI } = useAQI();
  const [sortBy, setSortBy] = useState('aqi');
  const [selectedCities, setSelectedCities] = useState([]);

  const sortedCities = useMemo(() => {
    const cities = [...currentAQI];
    switch (sortBy) {
      case 'aqi':
        return cities.sort((a, b) => b.aqi - a.aqi);
      case 'city':
        return cities.sort((a, b) => a.city.localeCompare(b.city));
      case 'pm25':
        return cities.sort((a, b) => (b.pm25 || 0) - (a.pm25 || 0));
      default:
        return cities;
    }
  }, [currentAQI, sortBy]);

  const chartData = useMemo(() => {
    const citiesToShow = selectedCities.length > 0 
      ? sortedCities.filter(city => selectedCities.includes(city.city))
      : sortedCities.slice(0, 10);

    return citiesToShow.map(city => ({
      city: city.city,
      AQI: city.aqi,
      PM2_5: city.pm25 || 0,
      PM10: city.pm10 || 0,
      category: city.category
    }));
  }, [sortedCities, selectedCities]);

  const toggleCitySelection = (cityName) => {
    setSelectedCities(prev => 
      prev.includes(cityName)
        ? prev.filter(city => city !== cityName)
        : [...prev, cityName]
    );
  };

  const getCategoryColor = (category) => {
    const colors = {
      'Good': '#10B981',
      'Satisfactory': '#F59E0B',
      'Moderate': '#F97316',
      'Poor': '#EF4444',
      'Very Poor': '#8B5CF6',
      'Severe': '#DC2626'
    };
    return colors[category] || '#6B7280';
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">City Comparison</h1>
        <div className="flex items-center space-x-4">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          >
            <option value="aqi">Sort by AQI (Worst First)</option>
            <option value="city">Sort by City Name</option>
            <option value="pm25">Sort by PM2.5</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">AQI Comparison</h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="city" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip 
                    formatter={(value, name) => [value, name === 'PM2_5' ? 'PM2.5' : name]}
                    labelFormatter={(label) => `City: ${label}`}
                  />
                  <Legend />
                  <Bar 
                    dataKey="AQI" 
                    fill="#3B82F6" 
                    name="AQI"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar 
                    dataKey="PM2_5" 
                    fill="#EF4444" 
                    name="PM2.5"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="font-semibold mb-3">City Selection</h3>
            <div className="max-h-96 overflow-y-auto space-y-2">
              {sortedCities.map(city => (
                <div
                  key={city.city}
                  className={`flex items-center justify-between p-2 rounded cursor-pointer transition-colors ${
                    selectedCities.includes(city.city) 
                      ? 'bg-blue-100 border border-blue-300' 
                      : 'hover:bg-gray-50'
                  }`}
                  onClick={() => toggleCitySelection(city.city)}
                >
                  <div className="flex items-center space-x-3">
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: getCategoryColor(city.category) }}
                    ></div>
                    <span className="font-medium">{city.city}</span>
                  </div>
                  <div className="text-right">
                    <div className="font-bold">{city.aqi}</div>
                    <div className="text-xs text-gray-500 capitalize">{city.category}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 text-sm text-gray-600">
              {selectedCities.length} cities selected
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="font-semibold mb-3">AQI Categories</h3>
            <div className="space-y-2 text-sm">
              {['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'].map(category => (
                <div key={category} className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: getCategoryColor(category) }}
                    ></div>
                    <span>{category}</span>
                  </div>
                  <span className="text-gray-600">
                    {currentAQI.filter(city => city.category === category).length} cities
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Detailed City Data</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  City
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  AQI
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Category
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  PM2.5
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  PM10
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Last Updated
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {sortedCities.map(city => (
                <tr key={city.city} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                    {city.city}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="font-bold">{city.aqi}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      city.category === 'Good' ? 'bg-green-100 text-green-800' :
                      city.category === 'Satisfactory' ? 'bg-yellow-100 text-yellow-800' :
                      city.category === 'Moderate' ? 'bg-orange-100 text-orange-800' :
                      city.category === 'Poor' ? 'bg-red-100 text-red-800' :
                      'bg-purple-100 text-purple-800'
                    }`}>
                      {city.category}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-500">
                    {city.pm25 || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-500">
                    {city.pm10 || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(city.timestamp).toLocaleTimeString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default CityComparison;