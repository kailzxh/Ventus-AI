// frontend/src/context/AQIContext.js
import React, { createContext, useContext, useState, useEffect } from 'react';
import { aqiService } from '../services/aqiService';

const AQIContext = createContext();

export const useAQI = () => {
  const context = useContext(AQIContext);
  if (!context) {
    throw new Error('useAQI must be used within an AQIProvider');
  }
  return context;
};

// City normalization function (export for reuse)
export const normalizeCityName = (city) => {
  const cityMap = {
    'new delhi': 'Delhi',
    'delhi': 'Delhi',
    'bengaluru': 'Bangalore',
    'bangalore': 'Bangalore',
    'mumbai': 'Mumbai',
    'chennai': 'Chennai',
    'kolkata': 'Kolkata',
    'hyderabad': 'Hyderabad',
    'ahmedabad': 'Ahmedabad',
    'pune': 'Pune',
    'surat': 'Surat',
    'jaipur': 'Jaipur',
    'lucknow': 'Lucknow',
    'kanpur': 'Kanpur',
    'nagpur': 'Nagpur',
    'indore': 'Indore',
    'thane': 'Thane',
    'bhopal': 'Bhopal',
    'visakhapatnam': 'Visakhapatnam',
    'patna': 'Patna',
    'vadodara': 'Vadodara',
    'ghaziabad': 'Ghaziabad',
    'ludhiana': 'Ludhiana',
    'agra': 'Agra',
    'nashik': 'Nashik',
    'faridabad': 'Faridabad'
  };
  return cityMap[city.toLowerCase()] || city;
};

export const AQIProvider = ({ children }) => {
  const [currentAQI, setCurrentAQI] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState(true);
  const [selectedCity, setSelectedCity] = useState('Delhi');
  const [availableCities, setAvailableCities] = useState(['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']);
  const [systemStatus, setSystemStatus] = useState({
    initialized: false,
    status: 'loading',
    error: null
  });
  const [apiError, setApiError] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [modelPerformance, setModelPerformance] = useState(null);

  // Initialize system and load initial data
  useEffect(() => {
    initializeSystem();
    const interval = setInterval(checkSystemStatus, 300000); // Check status every 5 minutes
    return () => clearInterval(interval);
  }, []);

  // Fetch current AQI data periodically
  useEffect(() => {
    if (systemStatus.initialized) {
      loadCurrentAQI();
      const interval = setInterval(loadCurrentAQI, 300000); // Refresh every 5 minutes
      return () => clearInterval(interval);
    }
  }, [systemStatus.initialized]);

  const initializeSystem = async () => {
    try {
      setLoading(true);
      setApiError(null);
      console.log('🚀 Initializing AQI system...');
      
      // Check if system needs initialization
      const status = await aqiService.getSystemStatus();
      console.log('📊 System status:', status);
      
      if (!status.system_initialized) {
        console.log('🔄 System not initialized, initializing now...');
        await aqiService.initializeSystem();
      }
      
      // Load initial data
      const [citiesResponse, modelsResponse, performanceResponse] = await Promise.all([
        aqiService.getAvailableCities(),
        aqiService.getAvailableModels(),
        aqiService.getModelPerformance()
      ]);

      if (process.env.NODE_ENV === 'development') {
        console.log('🏙️ Available cities:', citiesResponse);
        console.log('🤖 Available models:', modelsResponse);
        console.log('📊 Model performance:', performanceResponse);
      }
      
      // Extract cities - handle different response formats
      const cities = citiesResponse.cities || citiesResponse || [];
      const citiesList = Array.isArray(cities) ? cities : ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'];
      
      // Extract models - handle different response formats
      const models = modelsResponse.available_models || modelsResponse.models || modelsResponse || [];
      const modelsList = Array.isArray(models) ? models : [];
      
      setAvailableCities(citiesList);
      setAvailableModels(modelsList);
      setModelPerformance(performanceResponse || {});
      setSystemStatus({
        initialized: true,
        status: 'running',
        error: null
      });
      
      console.log('✅ System initialized successfully');
    } catch (error) {
      console.error('❌ Error initializing system:', error);
      setApiError(error.message || 'Failed to initialize system');
      setSystemStatus({
        initialized: false,
        status: 'error',
        error: error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const checkSystemStatus = async () => {
    try {
      setApiError(null);
      const [health, status] = await Promise.all([
        aqiService.healthCheck(),
        aqiService.getSystemStatus()
      ]);
      
      setSystemStatus({
        initialized: status.system_initialized,
        status: health.status,
        error: null
      });
    } catch (error) {
      console.error('Error checking system status:', error);
      setApiError(error.message || 'System status check failed');
      setSystemStatus(prev => ({
        ...prev,
        status: 'error',
        error: error.message
      }));
    }
  };

  const loadCurrentAQI = async () => {
    try {
      setLoading(true);
      setApiError(null);
      
      // Helper function to determine category from AQI
      const getCategoryFromAQI = (aqi) => {
        if (!aqi || isNaN(aqi)) return 'Unknown';
        if (aqi <= 50) return 'Good';
        if (aqi <= 100) return 'Satisfactory';
        if (aqi <= 200) return 'Moderate';
        if (aqi <= 300) return 'Poor';
        if (aqi <= 400) return 'Very Poor';
        return 'Severe';
      };

      // Helper function to normalize city data
      const normalizeCityData = (cityData, cityName = null) => {
        const name = cityName || cityData.city || cityData.name || cityData.City || 'Unknown';
        const aqi = typeof cityData.aqi === 'number' && !isNaN(cityData.aqi) 
          ? cityData.aqi 
          : (typeof cityData.AQI === 'number' && !isNaN(cityData.AQI))
          ? cityData.AQI
          : null;
        
        return {
          city: name,
          name: name,
          aqi: aqi,
          category: cityData.category || cityData.Category || (aqi ? getCategoryFromAQI(aqi) : 'Unknown'),
          pm25: cityData.pm25 || cityData.PM25 || cityData['PM2.5'] || null,
          pm10: cityData.pm10 || cityData.PM10 || null,
          timestamp: cityData.timestamp || cityData.Timestamp || new Date().toISOString(),
          predicted_aqi: cityData.predicted_aqi || cityData.predicted_AQI || null,
          accuracy: cityData.accuracy || null,
          trend: cityData.trend || null
        };
      };

      // Try cities/comparison endpoint first (faster, returns multiple cities)
      try {
        const comparisonData = await aqiService.getCityComparison();
        if (comparisonData && (comparisonData.cities || Array.isArray(comparisonData))) {
          const citiesData = comparisonData.cities || comparisonData;
          if (Array.isArray(citiesData) && citiesData.length > 0) {
            const normalizedCities = citiesData
              .filter(c => c && (c.city || c.name || c.City))
              .map(c => normalizeCityData(c));
            
            if (normalizedCities.length > 0) {
              setCurrentAQI(normalizedCities);
              return;
            }
          }
        }
      } catch (error) {
        if (process.env.NODE_ENV === 'development') {
          console.warn('Comparison endpoint failed, trying real-time:', error.message);
        }
      }

      // Try current-aqi endpoint
      try {
        const data = await aqiService.getCurrentAQI();
        const citiesData = (data && data.cities) ? data.cities : (Array.isArray(data) ? data : []);
        
        if (Array.isArray(citiesData) && citiesData.length > 0) {
          const normalizedCities = citiesData
            .filter(c => c && (c.city || c.name || c.City))
            .map(c => normalizeCityData(c));
          
          if (normalizedCities.length > 0) {
            setCurrentAQI(normalizedCities);
            return;
          }
        }
      } catch (error) {
        if (process.env.NODE_ENV === 'development') {
          console.warn('Current AQI endpoint failed, trying real-time:', error.message);
        }
      }

      // Fallback: Use real-time API for top cities
      try {
        const citiesResponse = await aqiService.getAvailableCities();
        const cities = citiesResponse.cities || citiesResponse || [];
        
        if (Array.isArray(cities) && cities.length > 0) {
          // Fetch real-time data for top 10 cities only (to avoid too many requests)
          const topCities = cities.slice(0, 10);
          const realtimePromises = topCities.map(async (cityName) => {
            try {
              const realtimeData = await aqiService.getRealTimeComparison(cityName);
              if (realtimeData && realtimeData.realtime) {
                const aqiValue = realtimeData.realtime.aqi;
                return {
                  city: cityName,
                  name: cityName,
                  aqi: typeof aqiValue === 'number' && !isNaN(aqiValue) ? aqiValue : null,
                  category: realtimeData.realtime.category || (aqiValue ? getCategoryFromAQI(aqiValue) : 'Unknown'),
                  pm25: realtimeData.realtime.pm25 || null,
                  pm10: realtimeData.realtime.pm10 || null,
                  timestamp: realtimeData.realtime.timestamp || new Date().toISOString(),
                  predicted_aqi: realtimeData.predictions?.today?.predicted_aqi || null,
                  accuracy: realtimeData.accuracy?.today || null,
                  trend: realtimeData.trend_analysis?.short_term || null,
                  // Store full realtime data for detailed display
                  realtimeData: realtimeData
                };
              }
              return null;
            } catch (error) {
              return null;
            }
          });

          const realtimeResults = await Promise.allSettled(realtimePromises);
          const validCities = realtimeResults
            .filter(result => result.status === 'fulfilled' && result.value !== null)
            .map(result => result.value);

          if (validCities.length > 0) {
            setCurrentAQI(validCities);
            return;
          }
        }
      } catch (error) {
        if (process.env.NODE_ENV === 'development') {
          console.warn('Real-time fallback failed:', error.message);
        }
      }

      // If all methods fail, set empty array
      setCurrentAQI([]);
    } catch (error) {
      console.error('Error loading current AQI:', error);
      setApiError(error.message || 'Failed to load current AQI data');
      setCurrentAQI([]);
    } finally {
      setLoading(false);
    }
  };

  const predictAQI = async (city, date, modelType = 'auto') => {
    try {
      // Backend handles city name normalization, so send as-is
      console.log(`🔮 Predicting AQI for: ${city}, Date: ${date}, Model: ${modelType}`);
      
      setApiError(null);
      const prediction = await aqiService.predictAQI(city, date, modelType);
      console.log('✅ Prediction result:', prediction);
      return prediction;
    } catch (error) {
      console.error('❌ Error predicting AQI:', error);
      setApiError(error.message || 'Prediction failed. Please try again.');
      throw error;
    }
  };

  const getFuturePredictions = async (city, days = 7) => {
    try {
      // Backend handles city name normalization
      console.log(`📅 Getting future predictions for: ${city}, Days: ${days}`);
      
      setApiError(null);
      const data = await aqiService.getFuturePredictions(city, days);
      setPredictions(prev => ({
        ...prev,
        [city]: data.predictions || data
      }));
      return data.predictions || data;
    } catch (error) {
      console.error('Error getting future predictions:', error);
      setApiError(error.message || 'Failed to get future predictions');
      throw error;
    }
  };

  const getStationPredictions = async (city, date, modelType = 'auto') => {
    try {
      // Backend handles city name normalization
      console.log(`🏭 Getting station predictions for: ${city}`);
      
      setApiError(null);
      const predictions = await aqiService.predictStationAQI(city, date, modelType);
      return predictions;
    } catch (error) {
      console.error('Error getting station predictions:', error);
      setApiError(error.message || 'Failed to get station predictions');
      return { predictions: [] }; // Return empty predictions instead of throwing
    }
  };

  const getCityStations = async (city) => {
    try {
      // Backend handles city name normalization
      console.log(`📍 Getting stations for: ${city}`);
      
      setApiError(null);
      const stations = await aqiService.getCityStations(city);
      return stations;
    } catch (error) {
      console.error('Error getting city stations:', error);
      setApiError(error.message || 'Failed to get city stations');
      throw error;
    }
  };

  const getRealTimeComparison = async (city) => {
    try {
      // Backend handles city name normalization
      if (process.env.NODE_ENV === 'development') {
        console.log(`🔍 Getting real-time comparison for: ${city}`);
      }
      
      setApiError(null);
      const comparison = await aqiService.getRealTimeComparison(city);
      
      // If null is returned (404), return empty data structure
      if (!comparison) {
        return { predictions: [], pollutants: [], realtime: null };
      }
      
      return comparison;
    } catch (error) {
      // Real-time data may not be available for all cities (returns 404)
      // This is expected behavior, so we don't set it as an error
      if (process.env.NODE_ENV === 'development') {
        console.warn('Real-time data not available for city:', city, error.message);
      }
      return { predictions: [], pollutants: [], realtime: null }; // Return empty data instead of throwing
    }
  };

  const getCityComparison = async () => {
    try {
      console.log('🏙️ Getting city comparison data');
      
      setApiError(null);
      
      // Try to get comparison endpoint first
      try {
        const comparison = await aqiService.getCityComparison();
        if (comparison && (comparison.cities || Array.isArray(comparison))) {
          return comparison;
        }
      } catch (error) {
        console.warn('Comparison endpoint failed, using real-time data:', error.message);
      }

      // Fallback: Use real-time data for available cities
      const citiesResponse = await aqiService.getAvailableCities();
      const cities = citiesResponse.cities || citiesResponse || [];
      
      if (!Array.isArray(cities) || cities.length === 0) {
        throw new Error('No cities available');
      }

      // Fetch real-time data for each city
      const realtimePromises = cities.slice(0, 15).map(async (cityName) => {
        try {
          const realtimeData = await aqiService.getRealTimeComparison(cityName);
          if (realtimeData && realtimeData.realtime) {
            return {
              city: cityName,
              name: cityName,
              aqi: realtimeData.realtime.aqi,
              category: realtimeData.realtime.category || 'Unknown',
              pm25: realtimeData.realtime.pm25,
              pm10: realtimeData.realtime.pm10,
              timestamp: realtimeData.realtime.timestamp || new Date().toISOString(),
              predicted_aqi: realtimeData.predictions?.today?.predicted_aqi,
              accuracy: realtimeData.accuracy?.today,
              trend: realtimeData.trend_analysis?.short_term,
              // Store full realtime data for detailed display
              realtimeData: realtimeData
            };
          }
          return null;
        } catch (error) {
          if (process.env.NODE_ENV === 'development') {
            console.warn(`Real-time data not available for ${cityName}`);
          }
          return null;
        }
      });

      const realtimeResults = await Promise.allSettled(realtimePromises);
      const validCities = realtimeResults
        .filter(result => result.status === 'fulfilled' && result.value !== null)
        .map(result => result.value);

      if (validCities.length === 0) {
        // Final fallback to current-aqi
        const data = await aqiService.getCurrentAQI();
        return {
          cities: (data && data.cities) ? data.cities : (Array.isArray(data) ? data : [])
        };
      }

      return { cities: validCities };
    } catch (error) {
      console.error('Error getting city comparison:', error);
      setApiError(error.message || 'Failed to get city comparison');
      throw error;
    }
  };

  const getCityDebug = async (city) => {
    try {
      console.log(`🔍 Getting debug info for: ${city}`);
      
      setApiError(null);
      const debug = await aqiService.getCityDebug(city);
      return debug;
    } catch (error) {
      console.error('Error getting city debug:', error);
      // Don't throw for debug endpoints, just log
      return null;
    }
  };

  const refreshData = async () => {
    await loadCurrentAQI();
    await checkSystemStatus();
  };

  const value = {
    // State
    currentAQI,
    predictions,
    loading,
    selectedCity,
    availableCities,
    systemStatus,
    apiError,
    availableModels,
    modelPerformance,

    // Setters
    setSelectedCity,

    // Actions
    refreshData,
    predictAQI,
    getFuturePredictions,
    getStationPredictions,
    getCityStations,
    getRealTimeComparison,
    getCityComparison,
    getCityDebug,
    initializeSystem,
    checkSystemStatus,
    normalizeCityName // Export the function for components
  };

  return (
    <AQIContext.Provider value={value}>
      {children}
    </AQIContext.Provider>
  );
};