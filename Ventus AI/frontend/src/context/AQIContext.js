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

export const AQIProvider = ({ children }) => {
  const [currentAQI, setCurrentAQI] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState(true);
  const [selectedCity, setSelectedCity] = useState('Delhi');
  const [availableCities, setAvailableCities] = useState(['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']);

  // Normalize city names to match API
  const normalizeCity = (city) => {
    const cityMap = {
      'New Delhi': 'Delhi',
      'Bengaluru': 'Bangalore',
    };
    return cityMap[city] || city;
  };
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
      // Check if system needs initialization
      const status = await aqiService.getSystemStatus();
      
      if (!status.system_initialized) {
        await aqiService.initializeSystem();
      }
      
      // Load initial data
      const [cities, models, performance] = await Promise.all([
        aqiService.getAvailableCities(),
        aqiService.getAvailableModels(),
        aqiService.getModelPerformance()
      ]);

      setAvailableCities(cities);
      setAvailableModels(models);
      setModelPerformance(performance);
      setSystemStatus({
        initialized: true,
        status: 'running',
        error: null
      });
    } catch (error) {
      console.error('Error initializing system:', error);
      setApiError(error);
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
      setApiError(error);
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
      const data = await aqiService.getCurrentAQI();
      setCurrentAQI((data && data.cities) ? data.cities : []);
    } catch (error) {
      console.error('Error loading current AQI:', error);
      setApiError(error);
    } finally {
      setLoading(false);
    }
  };

  const predictAQI = async (city, date, modelType = 'nf_vae') => {
    try {
      const normalizedCity = normalizeCity(city);
      setApiError(null);
      const prediction = await aqiService.predictAQI(normalizedCity, date, modelType);
      return prediction;
    } catch (error) {
      console.error('Error predicting AQI:', error);
      setApiError(error);
      throw error;
    }
  };

  const getFuturePredictions = async (city, days = 7) => {
    try {
      setApiError(null);
      const data = await aqiService.getFuturePredictions(city, days);
      setPredictions(prev => ({
        ...prev,
        [city]: data.predictions
      }));
      return data.predictions;
    } catch (error) {
      console.error('Error getting future predictions:', error);
      setApiError(error);
      throw error;
    }
  };

  const getStationPredictions = async (city, date, modelType = 'nf_vae') => {
    try {
      const normalizedCity = normalizeCity(city);
      setApiError(null);
      const predictions = await aqiService.predictStationAQI(normalizedCity, date, modelType);
      return predictions;
    } catch (error) {
      console.error('Error getting station predictions:', error);
      setApiError(error);
      return { predictions: [] }; // Return empty predictions instead of throwing
    }
  };

  const getCityStations = async (city) => {
    try {
      setApiError(null);
      const stations = await aqiService.getCityStations(city);
      return stations;
    } catch (error) {
      console.error('Error getting city stations:', error);
      setApiError(error);
      throw error;
    }
  };

  const getRealTimeComparison = async (city) => {
    try {
      const normalizedCity = normalizeCity(city);
      setApiError(null);
      const comparison = await aqiService.getRealTimeComparison(normalizedCity);
      return comparison;
    } catch (error) {
      console.error('Error getting real-time comparison:', error);
      setApiError(error);
      return { predictions: [], pollutants: [] }; // Return empty data instead of throwing
    }
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
    refreshData: loadCurrentAQI,
    predictAQI,
    getFuturePredictions,
    getStationPredictions,
    getCityStations,
    getRealTimeComparison,
    initializeSystem,
    checkSystemStatus
  };

  return (
    <AQIContext.Provider value={value}>
      {children}
    </AQIContext.Provider>
  );
};