// frontend/src/services/aqiService.js
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

class AQIService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const method = options.method || 'GET';
    const hasBody = options.body && (method === 'POST' || method === 'PUT' || method === 'PATCH');
    
    const config = {
      method: method,
      headers: {
        ...(hasBody && { 'Content-Type': 'application/json' }),
        ...options.headers,
      },
    };

    // Only add body for POST/PUT/PATCH requests
    if (hasBody) {
      config.body = options.body;
    }

    try {
      console.log(`🌐 API Call: ${method} ${url}`, config);
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.error || `HTTP ${response.status}: ${response.statusText}`;
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      console.log(`✅ API Response from ${endpoint}:`, data);
      return data;
    } catch (error) {
      console.error(`❌ API Error at ${endpoint}:`, error);
      throw error;
    }
  }

  // Health check
  async healthCheck() {
    return this.request('/health');
  }

  // System status
  async getSystemStatus() {
    return this.request('/status');
  }

  // Initialize system
  async initializeSystem() {
    return this.request('/initialize', {
      method: 'POST',
    });
  }

  // Get available cities
  async getAvailableCities() {
    return this.request('/cities');
  }

  // Get current AQI for all cities
  async getCurrentAQI() {
    return this.request('/current-aqi');
  }

  // Predict AQI
  async predictAQI(city, date, modelType = 'auto') {
    return this.request('/predict', {
      method: 'POST',
      body: JSON.stringify({
        city,
        date,
        model_type: modelType,
      }),
    });
  }

  // Get future predictions
  async getFuturePredictions(city, days = 7) {
    return this.request(`/predict/${encodeURIComponent(city)}/future?days=${days}`);
  }

  // Predict station AQI
  async predictStationAQI(city, date, modelType = 'auto') {
    return this.request('/predict/stations', {
      method: 'POST',
      body: JSON.stringify({
        city,
        date,
        model_type: modelType,
        include_city_level: true,
      }),
    });
  }

  // Get city stations
  async getCityStations(city) {
    return this.request(`/cities/${encodeURIComponent(city)}/stations`);
  }

  // Get real-time comparison
  async getRealTimeComparison(city) {
    try {
      const data = await this.request(`/realtime/${encodeURIComponent(city)}`);
      return data;
    } catch (error) {
      // Real-time endpoint may return 404 for some cities - this is expected
      // Don't throw error, just return null so caller can handle gracefully
      if (error.message && (error.message.includes('404') || error.message.includes('No real-time data'))) {
        return null;
      }
      // For other errors, still return null but log it
      console.warn(`Real-time API error for ${city}:`, error.message);
      return null;
    }
  }

  // Get available models
  async getAvailableModels() {
    return this.request('/models/available');
  }

  // Get model performance
  async getModelPerformance() {
    return this.request('/models/performance');
  }

  // Get city comparison data
  async getCityComparison() {
    return this.request('/cities/comparison');
  }

  // Get debug info for a city
  async getCityDebug(city) {
    return this.request(`/debug/${encodeURIComponent(city)}`);
  }

  // Predict AQI (GET method - alternative)
  async predictAQIGet(city, date, modelType = 'auto') {
    const params = new URLSearchParams({
      city: city,
      date: date,
      model_type: modelType
    });
    return this.request(`/predict?${params.toString()}`);
  }
}

export const aqiService = new AQIService();