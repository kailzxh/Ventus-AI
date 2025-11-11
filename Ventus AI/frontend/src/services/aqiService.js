// frontend/src/services/aqiService.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  // Add retry logic
  retry: 3,
  retryDelay: 1000,
  // Handle timeouts more gracefully
  timeoutErrorMessage: 'Request timeout - server might be busy'
});

export const aqiService = {
  // System Status and Health
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  async getSystemStatus() {
    const response = await api.get('/status');
    return response.data;
  },

  async initializeSystem() {
    const response = await api.post('/initialize');
    return response.data;
  },

  // City and Station Data
  async getAvailableCities() {
    const response = await api.get('/cities');
    return response.data;
  },

  async getCurrentAQI() {
    const response = await api.get('/current-aqi');
    return response.data;
  },

  async getCityStations(city) {
    const response = await api.get(`/cities/${city}/stations`);
    return response.data;
  },

  async getCityComparison() {
    const response = await api.get('/cities/comparison');
    return response.data;
  },

  // Predictions and Real-time Data
  async predictAQI(city, date, modelType = 'nf_vae') {
    const response = await api.post('/predict', {
      city,
      date,
      model_type: modelType
    });
    return response.data;
  },

  async getFuturePredictions(city, days = 7) {
    const response = await api.get(`/predict/${city}/future?days=${days}`);
    return response.data;
  },

  async predictStationAQI(city, date, modelType = 'nf_vae') {
    const response = await api.post('/predict/stations', {
      city,
      date,
      model_type: modelType
    });
    return response.data;
  },

  async getRealTimeComparison(city) {
    const response = await api.get(`/realtime/${city}`);
    return response.data;
  },

  // Model Information
  async getAvailableModels() {
    const response = await api.get('/models/available');
    return response.data;
  },

  async getModelPerformance() {
    const response = await api.get('/models/performance');
    return response.data;
  }
};

// Error handler
const errorHandler = (error) => {
  // Normalize error into { status, message, details }
  try {
    if (error.response) {
      const status = error.response.status || 500;
      const contentType = error.response.headers && error.response.headers['content-type'] ? error.response.headers['content-type'] : '';

      // If server returned HTML (Werkzeug debugger) or plain text, capture a short snippet
      if (typeof error.response.data === 'string' && (contentType.includes('text/html') || error.response.data.trim().startsWith('<'))) {
        const snippet = error.response.data.slice(0, 200);
        console.error('API Error HTML Response snippet:', snippet);
        throw { status, message: 'Server error (non-JSON response)', details: { raw: snippet } };
      }

      // If JSON-like response, attempt to read message
      const data = error.response.data || {};
      const message = (data && (data.message || data.error)) || 'An error occurred';
      throw { status, message, details: data };
    } else if (error.request) {
      console.error('API Request Error: No response received');
      throw { status: 503, message: 'Service unavailable', details: 'No response from server' };
    } else {
      console.error('API Setup Error:', error.message);
      throw { status: 500, message: 'Request failed', details: error.message };
    }
  } catch (e) {
    // If our normalization throws, ensure we reject with a sensible object
    if (e && e.status && e.message) throw e;
    throw { status: 500, message: 'Unknown API error', details: String(e || error) };
  }
};

// Response interceptor
api.interceptors.response.use(
  (response) => {
    // If server returned HTML on success (unlikely) treat as error
    const contentType = response.headers && response.headers['content-type'] ? response.headers['content-type'] : '';
    if (typeof response.data === 'string' && (contentType.includes('text/html') || response.data.trim().startsWith('<'))) {
      const snippet = response.data.slice(0, 200);
      return Promise.reject({ response: { status: 500, data: snippet, headers: response.headers } });
    }
    return response;
  },
  (error) => errorHandler(error)
);

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // You can add auth headers or other request modifications here
    return config;
  },
  (error) => {
    console.error('Request Error:', error);
    return Promise.reject(error);
  }
);