// frontend/src/pages/Dashboard.js
import React, { useState, useEffect } from 'react';
import { useAQI } from '../context/AQIContext';
import AQICard from '../components/AQICard';
import AQITrendChart from '../components/AQITrendChart';
import CityMap from '../components/CityMap';
import QuickStats from '../components/QuickStats';
import RealTimeCityCard from '../components/RealTimeCityCard';
import PredictionCard from '../components/PredictionCard';
import LoadingSpinner from '../components/LoadingSpinner';
import EmptyState from '../components/EmptyState';
import { AlertCircle, RefreshCw, Activity, TrendingUp, Calendar } from 'lucide-react';

// List of cities to monitor
const MONITORED_CITIES = ['delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore', 'hyderabad'];

const Dashboard = () => {
  const { currentAQI, loading, apiError, refreshData, systemStatus } = useAQI();
  const [selectedCityForDetails, setSelectedCityForDetails] = useState(null);
  const [detailedCityData, setDetailedCityData] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [realtimeData, setRealtimeData] = useState([]);
  const [loadingRealtime, setLoadingRealtime] = useState(false);

  // Fetch real-time data for all monitored cities
  useEffect(() => {
    fetchAllRealtimeData();
  }, []);

  // Fetch real-time data for a specific city
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

  // Fetch real-time data for all monitored cities
  const fetchAllRealtimeData = async () => {
    setLoadingRealtime(true);
    try {
      const promises = MONITORED_CITIES.map(city => fetchRealtimeCityData(city));
      const results = await Promise.all(promises);
      
      const validResults = results.filter(result => 
        result && result.realtime && result.realtime.aqi
      );
      
      setRealtimeData(validResults);
    } catch (error) {
      console.error('Error fetching all realtime data:', error);
    } finally {
      setLoadingRealtime(false);
    }
  };

  // Fetch detailed city data when selection changes
  useEffect(() => {
    if (selectedCityForDetails) {
      fetchCityDetails(selectedCityForDetails);
    }
  }, [selectedCityForDetails]);

  const fetchCityDetails = async (cityName) => {
    setLoadingDetails(true);
    try {
      const data = await fetchRealtimeCityData(cityName);
      if (data) {
        setDetailedCityData(data);
      }
    } catch (error) {
      console.error('Error fetching city details:', error);
    } finally {
      setLoadingDetails(false);
    }
  };

  // Transform API data for frontend components
  const transformCityData = (cityData) => {
    if (!cityData) return null;
    
    return {
      city: cityData.city || cityData.city_requested,
      aqi: cityData.realtime?.aqi,
      category: cityData.realtime?.category,
      pm25: cityData.realtime?.pm25,
      pm10: cityData.realtime?.pm10,
      timestamp: cityData.realtime?.timestamp,
      source: cityData.realtime?.source,
      // Add prediction data
      predictions: cityData.predictions,
      accuracy: cityData.accuracy,
      health_advice: cityData.health_advice,
      trend_analysis: cityData.trend_analysis
    };
  };

  // Combine data from context and real-time API calls
  const getCombinedCityData = () => {
    const contextCities = Array.isArray(currentAQI) 
      ? currentAQI
          .filter(c => c && (c.city || c.name))
          .map(city => transformCityData(city))
          .filter(Boolean)
      : [];

    const realtimeCities = realtimeData
      .map(city => transformCityData(city))
      .filter(Boolean);

    // Merge arrays, preferring real-time data when available
    const mergedCities = [...contextCities];
    
    realtimeCities.forEach(realtimeCity => {
      const existingIndex = mergedCities.findIndex(c => 
        c.city.toLowerCase() === realtimeCity.city.toLowerCase()
      );
      
      if (existingIndex >= 0) {
        // Update with real-time data if it's more recent or has AQI
        if (realtimeCity.aqi && (!mergedCities[existingIndex].aqi || realtimeCity.timestamp > mergedCities[existingIndex].timestamp)) {
          mergedCities[existingIndex] = realtimeCity;
        }
      } else {
        mergedCities.push(realtimeCity);
      }
    });

    return mergedCities;
  };

  const validCities = getCombinedCityData();

  // Sort cities by AQI (worst first)
  const worstCities = validCities
    .filter(c => typeof c.aqi === 'number' && !isNaN(c.aqi))
    .sort((a, b) => (b.aqi || 0) - (a.aqi || 0))
    .slice(0, 6);

  // Get cities with realtime data
  const citiesWithRealtimeData = validCities.filter(c => c.aqi && c.category);

  // Get selected city's detailed data
  const selectedCityDetails = selectedCityForDetails 
    ? (detailedCityData ? transformCityData(detailedCityData) : citiesWithRealtimeData.find(c => c.city === selectedCityForDetails))
    : citiesWithRealtimeData[0];

  // Refresh all data
  const handleRefresh = async () => {
    await refreshData();
    await fetchAllRealtimeData();
    if (selectedCityForDetails) {
      await fetchCityDetails(selectedCityForDetails);
    }
  };

  const isLoading = loading || loadingRealtime;

  if (isLoading && validCities.length === 0) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">AQI Dashboard</h1>
            <p className="text-gray-600">Real-time Air Quality Monitoring</p>
          </div>
        </div>
        <LoadingSpinner message="Loading dashboard data..." size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">AQI Dashboard</h1>
          <p className="text-gray-600 mt-1">Real-time Air Quality Monitoring & Predictions</p>
        </div>
        <div className="flex items-center space-x-3">
          {systemStatus && (
            <div className="flex items-center text-sm">
              <div className={`w-2 h-2 rounded-full mr-2 ${
                systemStatus.initialized && systemStatus.status === 'running' 
                  ? 'bg-green-500' 
                  : 'bg-yellow-500'
              }`}></div>
              <span className="text-gray-600">
                {systemStatus.initialized && systemStatus.status === 'running' 
                  ? 'System Online' 
                  : 'System Loading'}
              </span>
            </div>
          )}
          <button
            onClick={handleRefresh}
            disabled={isLoading}
            className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 disabled:opacity-50 transition-colors"
          >
            <RefreshCw size={16} className={`mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {apiError && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 flex items-start">
          <AlertCircle size={20} className="text-red-400 mr-3 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h3 className="text-sm font-medium text-red-800">Error Loading Data</h3>
            <p className="text-sm text-red-700 mt-1">{apiError}</p>
            <button
              onClick={handleRefresh}
              className="mt-2 text-sm text-red-600 hover:text-red-800 underline"
            >
              Try again
            </button>
          </div>
        </div>
      )}

      {validCities.length === 0 && !isLoading && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
          <EmptyState
            icon={AlertCircle}
            title="No city data available"
            message="Unable to load city data. Please check your connection and try refreshing."
            action={handleRefresh}
            actionLabel="Refresh Data"
          />
        </div>
      )}

      {validCities.length > 0 && (
        <>
          <QuickStats cities={validCities} />

          {/* Real-time Detailed View with Predictions */}
          {selectedCityDetails && (
            <div className="space-y-6">
              {/* Current AQI Card */}
              <div className="bg-white rounded-lg shadow-lg">
                <div className="p-4 border-b border-gray-200 flex items-center justify-between">
                  <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                    <Activity className="mr-2" size={24} />
                    Current Air Quality - {selectedCityDetails.city}
                  </h2>
                  {citiesWithRealtimeData.length > 1 && (
                    <select
                      value={selectedCityForDetails || citiesWithRealtimeData[0]?.city}
                      onChange={(e) => setSelectedCityForDetails(e.target.value)}
                      className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
                    >
                      {citiesWithRealtimeData.map((city) => (
                        <option key={city.city} value={city.city}>
                          {city.city}
                        </option>
                      ))}
                    </select>
                  )}
                </div>
                <div className="p-6">
                  <RealTimeCityCard 
                    realtimeData={selectedCityDetails}
                    cityName={selectedCityDetails.city}
                  />
                </div>
              </div>

              {/* Predictions Section */}
              {selectedCityDetails.predictions && (
                <div className="bg-white rounded-lg shadow-lg">
                  <div className="p-4 border-b border-gray-200">
                    <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                      <TrendingUp className="mr-2" size={24} />
                      AQI Predictions - {selectedCityDetails.city}
                    </h2>
                    {selectedCityDetails.accuracy && (
                      <p className="text-sm text-gray-600 mt-1">
                        Model Accuracy: <span className="font-medium text-green-600">
                          {selectedCityDetails.accuracy.today}% ({selectedCityDetails.accuracy.status})
                        </span>
                      </p>
                    )}
                  </div>
                  <div className="p-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {selectedCityDetails.predictions.today && (
                        <PredictionCard
                          prediction={selectedCityDetails.predictions.today}
                          title="Today"
                          icon={Calendar}
                        />
                      )}
                      {selectedCityDetails.predictions.tomorrow && (
                        <PredictionCard
                          prediction={selectedCityDetails.predictions.tomorrow}
                          title="Tomorrow"
                          icon={Calendar}
                        />
                      )}
                      {selectedCityDetails.predictions.next_week && (
                        <PredictionCard
                          prediction={selectedCityDetails.predictions.next_week}
                          title="Next Week"
                          icon={TrendingUp}
                        />
                      )}
                    </div>

                    {/* Trend Analysis */}
                    {selectedCityDetails.trend_analysis && (
                      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">Trend Analysis</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="font-medium">Short-term trend:</span>{' '}
                            <span className={`font-semibold ${
                              selectedCityDetails.trend_analysis.short_term === 'improving' 
                                ? 'text-green-600' 
                                : selectedCityDetails.trend_analysis.short_term === 'worsening'
                                ? 'text-red-600'
                                : 'text-yellow-600'
                            }`}>
                              {selectedCityDetails.trend_analysis.short_term}
                            </span>
                          </div>
                          <div>
                            <span className="font-medium">Long-term trend:</span>{' '}
                            <span className={`font-semibold ${
                              selectedCityDetails.trend_analysis.long_term === 'improving' 
                                ? 'text-green-600' 
                                : selectedCityDetails.trend_analysis.long_term === 'worsening'
                                ? 'text-red-600'
                                : 'text-yellow-600'
                            }`}>
                              {selectedCityDetails.trend_analysis.long_term}
                            </span>
                          </div>
                          <div>
                            <span className="font-medium">Confidence:</span>{' '}
                            <span className="font-semibold text-blue-600">
                              {selectedCityDetails.trend_analysis.confidence}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Health Advice */}
                    {selectedCityDetails.health_advice && (
                      <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">Health Advice</h3>
                        <p className="text-sm text-gray-700">{selectedCityDetails.health_advice}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Main Dashboard Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-6">
              {/* Worst Affected Cities */}
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4 text-gray-900">Worst Affected Cities</h2>
                {worstCities.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {worstCities.map((city, index) => (
                      <div 
                        key={city.city || index}
                        onClick={() => {
                          setSelectedCityForDetails(city.city);
                          window.scrollTo({ top: 0, behavior: 'smooth' });
                        }}
                        className="cursor-pointer hover:opacity-80 transition-opacity"
                      >
                        <AQICard city={city} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    title="No AQI data available"
                    message="AQI data for cities will appear here once available."
                  />
                )}
              </div>

              {/* AQI Trends */}
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4 text-gray-900">AQI Trends</h2>
                <AQITrendChart cities={worstCities.slice(0, 3)} />
              </div>
            </div>

            {/* City Map */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4 text-gray-900">City Overview</h2>
              <CityMap cities={validCities} />
            </div>
          </div>
        </>
      )}

      {loadingDetails && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <LoadingSpinner message="Loading city details..." />
        </div>
      )}
    </div>
  );
};

export default Dashboard;