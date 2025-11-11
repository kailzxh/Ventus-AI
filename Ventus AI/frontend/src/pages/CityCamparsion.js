// frontend/src/pages/CityComparison.js
import React, { useState, useMemo, useEffect } from 'react';
import { useAQI } from '../context/AQIContext';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { RefreshCw, AlertCircle, Map } from 'lucide-react';
import LoadingSpinner from '../components/LoadingSpinner';
import EmptyState from '../components/EmptyState';

// List of cities to monitor for comparison
const COMPARISON_CITIES = ['delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore', 'hyderabad', 'pune', 'ahmedabad', 'jaipur', 'lucknow'];

const CityComparison = () => {
  const { currentAQI, getCityComparison, loading, apiError } = useAQI();
  const [sortBy, setSortBy] = useState('aqi');
  const [selectedCities, setSelectedCities] = useState([]);
  const [comparisonData, setComparisonData] = useState([]);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [directApiData, setDirectApiData] = useState([]);
  const [apiLoading, setApiLoading] = useState(false);

  // Fetch real-time data for all comparison cities
  const fetchRealtimeCityData = async (cityName) => {
    try {
      const response = await fetch(`http://localhost:5000/api/realtime/${cityName.toLowerCase()}`);
      if (response.ok) {
        const data = await response.json();
        return data;
      }
      return null;
    } catch (error) {
      console.error(`Error fetching realtime data for ${cityName}:`, error);
      return null;
    }
  };

  // Fetch real-time data for all comparison cities
  const fetchAllComparisonData = async () => {
    setApiLoading(true);
    try {
      const promises = COMPARISON_CITIES.map(city => fetchRealtimeCityData(city));
      const results = await Promise.all(promises);
      
      const validResults = results
        .filter(result => result && result.realtime && result.realtime.aqi)
        .map(result => transformCityData(result))
        .filter(Boolean);
      
      setDirectApiData(validResults);
      return validResults;
    } catch (error) {
      console.error('Error fetching comparison data:', error);
      return [];
    } finally {
      setApiLoading(false);
    }
  };

  // Transform API data for frontend components
  const transformCityData = (cityData) => {
    if (!cityData) return null;
    
    const cityName = cityData.city || cityData.city_requested;
    if (!cityName) return null;

    return {
      city: cityName,
      name: cityName,
      aqi: cityData.realtime?.aqi,
      category: cityData.realtime?.category,
      pm25: cityData.realtime?.pm25,
      pm10: cityData.realtime?.pm10,
      timestamp: cityData.realtime?.timestamp || new Date().toISOString(),
      // Add prediction data if available
      predicted_aqi: cityData.predictions?.today?.predicted_aqi,
      accuracy: cityData.accuracy?.today,
      trend: cityData.trend_analysis?.short_term,
      source: 'realtime-api'
    };
  };

  // Load comparison data on mount
  useEffect(() => {
    loadComparisonData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load all data sources
  const loadComparisonData = async () => {
    setComparisonLoading(true);
    try {
      // Fetch data from multiple sources in parallel
      const [directData, contextComparisonData] = await Promise.all([
        fetchAllComparisonData(),
        getContextComparisonData()
      ]);

      // Merge and deduplicate data
      const mergedData = mergeAndDeduplicateData(directData, contextComparisonData);
      
      setComparisonData(mergedData);

      if (process.env.NODE_ENV === 'development') {
        console.log('✅ Final comparison data:', mergedData);
      }
    } catch (error) {
      console.error('❌ Error loading comparison data:', error);
      // Fallback to direct API data or currentAQI
      if (directApiData.length > 0) {
        setComparisonData(directApiData);
      } else if (currentAQI && currentAQI.length > 0) {
        setComparisonData(currentAQI);
      }
    } finally {
      setComparisonLoading(false);
    }
  };

  // Get comparison data from context with fallback
  const getContextComparisonData = async () => {
    try {
      const data = await getCityComparison();
      
      // Handle different response formats
      let citiesData = [];
      if (data && data.cities && Array.isArray(data.cities)) {
        citiesData = data.cities;
      } else if (Array.isArray(data)) {
        citiesData = data;
      } else if (data && typeof data === 'object') {
        // Try to extract cities from object
        const keys = Object.keys(data);
        if (keys.length > 0 && Array.isArray(data[keys[0]])) {
          citiesData = data[keys[0]];
        }
      }

      return citiesData
        .filter(city => city && (city.city || city.name))
        .map(city => {
          const source = city.realtimeData || city;
          const cityName = source.city || source.name || city.city || city.name;
          
          return {
            city: cityName,
            name: cityName,
            aqi: typeof source.aqi === 'number' && !isNaN(source.aqi) ? source.aqi : (source.realtime?.aqi ?? null),
            category: source.category || source.realtime?.category || 'Unknown',
            pm25: source.pm25 || source.realtime?.pm25 || null,
            pm10: source.pm10 || source.realtime?.pm10 || null,
            timestamp: source.timestamp || new Date().toISOString(),
            predicted_aqi: source.predicted_aqi || source.predictions?.today?.predicted_aqi || null,
            accuracy: source.accuracy || source.accuracy?.today || null,
            trend: source.trend || source.trend_analysis?.short_term || null,
            source: 'context'
          };
        });
    } catch (error) {
      console.error('Error getting context comparison data:', error);
      return [];
    }
  };

  // Merge and deduplicate data from multiple sources
  const mergeAndDeduplicateData = (directData, contextData) => {
    const cityMap = new Map();

    // Helper function to add city to map
    const addCityToMap = (city) => {
      if (!city || !city.city) return;
      
      const key = city.city.toLowerCase();
      const existing = cityMap.get(key);
      
      // Prefer direct API data over context data
      if (!existing || city.source === 'realtime-api') {
        cityMap.set(key, city);
      }
    };

    // Add all cities from both sources
    [...directData, ...contextData].forEach(addCityToMap);

    // Convert map back to array and sort by AQI (worst first)
    return Array.from(cityMap.values())
      .filter(city => city.aqi && !isNaN(city.aqi))
      .sort((a, b) => (b.aqi || 0) - (a.aqi || 0));
  };

  // Use comparison data if available, otherwise use combined fallback
  const citiesToCompare = comparisonData.length > 0 
    ? comparisonData 
    : [...directApiData, ...currentAQI].filter(city => city && city.city);

  const sortedCities = useMemo(() => {
    const cities = [...citiesToCompare];
    switch (sortBy) {
      case 'aqi':
        return cities.sort((a, b) => (b.aqi || 0) - (a.aqi || 0));
      case 'city':
        return cities.sort((a, b) => (a.city || '').localeCompare(b.city || ''));
      case 'pm25':
        return cities.sort((a, b) => (b.pm25 || 0) - (a.pm25 || 0));
      default:
        return cities;
    }
  }, [citiesToCompare, sortBy]);

  const chartData = useMemo(() => {
    const citiesToShow = selectedCities.length > 0 
      ? sortedCities.filter(city => selectedCities.includes(city.city))
      : sortedCities.slice(0, 8); // Show top 8 cities by default

    return citiesToShow.map(city => ({
      city: city.city,
      AQI: city.aqi || 0,
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

  const selectAllCities = () => {
    setSelectedCities(sortedCities.map(city => city.city));
  };

  const clearSelection = () => {
    setSelectedCities([]);
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

  const isLoading = comparisonLoading || apiLoading;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">City Comparison</h1>
          <p className="text-gray-600 mt-1">Compare real-time AQI across multiple cities</p>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={loadComparisonData}
            disabled={isLoading}
            className="flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 disabled:opacity-50 transition-colors"
          >
            <RefreshCw size={16} className={`mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
          >
            <option value="aqi">Sort by AQI (Worst First)</option>
            <option value="city">Sort by City Name</option>
            <option value="pm25">Sort by PM2.5</option>
          </select>
        </div>
      </div>

      {apiError && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 flex items-start">
          <AlertCircle size={20} className="text-red-400 mr-3 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-red-800">Error Loading Data</h3>
            <p className="text-sm text-red-700 mt-1">{apiError}</p>
            <button
              onClick={loadComparisonData}
              className="mt-2 text-sm text-red-600 hover:text-red-800 underline"
            >
              Try again
            </button>
          </div>
        </div>
      )}

      {isLoading && citiesToCompare.length === 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <LoadingSpinner message="Loading city comparison data..." />
        </div>
      )}

      {!isLoading && citiesToCompare.length === 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <EmptyState
            icon={Map}
            title="No comparison data available"
            message="Unable to load city comparison data. Please try refreshing or check your connection."
            action={loadComparisonData}
            actionLabel="Refresh Data"
          />
        </div>
      )}

      {citiesToCompare.length > 0 && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold">AQI Comparison</h2>
                  <div className="text-sm text-gray-500">
                    {selectedCities.length > 0 
                      ? `${selectedCities.length} cities selected` 
                      : 'Showing top 8 cities by AQI'
                    }
                  </div>
                </div>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="city" 
                        angle={-45} 
                        textAnchor="end" 
                        height={80}
                        interval={0}
                      />
                      <YAxis />
                      <Tooltip 
                        formatter={(value, name) => [Math.round(value * 100) / 100, name === 'PM2_5' ? 'PM2.5' : name]}
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
                <div className="flex justify-between items-center mb-3">
                  <h3 className="font-semibold">City Selection</h3>
                  <div className="flex space-x-2">
                    <button
                      onClick={selectAllCities}
                      className="text-xs text-blue-600 hover:text-blue-800"
                    >
                      Select All
                    </button>
                    <button
                      onClick={clearSelection}
                      className="text-xs text-gray-600 hover:text-gray-800"
                    >
                      Clear
                    </button>
                  </div>
                </div>
                <div className="max-h-96 overflow-y-auto space-y-2">
                  {sortedCities.map(city => {
                    const cityName = city.city;
                    return (
                      <div
                        key={cityName}
                        className={`flex items-center justify-between p-2 rounded cursor-pointer transition-colors ${
                          selectedCities.includes(cityName) 
                            ? 'bg-blue-100 border border-blue-300' 
                            : 'hover:bg-gray-50'
                        }`}
                        onClick={() => toggleCitySelection(cityName)}
                      >
                        <div className="flex items-center space-x-3">
                          <div 
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: getCategoryColor(city.category) }}
                          ></div>
                          <span className="font-medium text-sm">{cityName}</span>
                        </div>
                        <div className="text-right">
                          <div className="font-bold">{city.aqi || '—'}</div>
                          <div className="text-xs text-gray-500 capitalize">
                            {city.category || 'Unknown'}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="font-semibold mb-3">AQI Distribution</h3>
                <div className="space-y-2 text-sm">
                  {['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'].map(category => {
                    const count = citiesToCompare.filter(city => city.category === category).length;
                    return (
                      <div key={category} className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <div 
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: getCategoryColor(category) }}
                          ></div>
                          <span>{category}</span>
                        </div>
                        <span className="text-gray-600">
                          {count} {count === 1 ? 'city' : 'cities'}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">Total cities:</span>
                    <span className="text-gray-600">{citiesToCompare.length}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6 mt-6">
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
                  {sortedCities.map(city => {
                    const cityName = city.city;
                    const timestamp = city.timestamp ? new Date(city.timestamp) : new Date();
                    return (
                      <tr key={cityName} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                          {cityName}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="font-bold">{city.aqi || '—'}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            city.category === 'Good' ? 'bg-green-100 text-green-800' :
                            city.category === 'Satisfactory' ? 'bg-yellow-100 text-yellow-800' :
                            city.category === 'Moderate' ? 'bg-orange-100 text-orange-800' :
                            city.category === 'Poor' ? 'bg-red-100 text-red-800' :
                            city.category === 'Very Poor' ? 'bg-purple-100 text-purple-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {city.category || 'Unknown'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-gray-500">
                          {city.pm25 !== undefined && city.pm25 !== null ? Math.round(city.pm25 * 10) / 10 : 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-gray-500">
                          {city.pm10 !== undefined && city.pm10 !== null ? Math.round(city.pm10 * 10) / 10 : 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {timestamp.toLocaleTimeString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default CityComparison;