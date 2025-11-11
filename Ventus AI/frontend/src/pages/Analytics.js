/* eslint-disable react-hooks/exhaustive-deps */
import React, { useEffect, useState } from 'react';
import { useAQI } from '../context/AQIContext';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, ScatterChart, Scatter
} from 'recharts';
import { TrendingUp, BarChart2, PieChart as PieIcon, Activity, Map, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';
import LoadingSpinner from '../components/LoadingSpinner';

const categoryColors = ['#10B981', '#F59E0B', '#F97316', '#EF4444', '#8B5CF6', '#DC2626'];

const Analytics = () => {
  const {
    systemStatus,
    availableModels = [],
    modelPerformance = {},
    getStationPredictions,
    getRealTimeComparison,
    selectedCity: contextSelectedCity,
    availableCities = [],
    setSelectedCity
  } = useAQI();

  const initialAnalyticsState = {
    modelMetrics: [],
    realTimeComparison: [],
    stationPredictions: [],
    categoryDistribution: [],
    pollutantTrends: []
  };

  const [analyticsData, setAnalyticsData] = useState(initialAnalyticsState);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedModel, setSelectedModel] = useState(availableModels[0] || 'nf_vae');
  const [dateRange, setDateRange] = useState('7d');
  const [error, setError] = useState(null);
  const [selectedCity, setSelectedCityLocal] = useState(contextSelectedCity || availableCities[0] || 'Delhi');

  // Update local selectedCity when context changes
  useEffect(() => {
    if (contextSelectedCity && contextSelectedCity !== selectedCity) {
      setSelectedCityLocal(contextSelectedCity);
    }
  }, [contextSelectedCity]);

  useEffect(() => {
    // Only load data if we have a valid city and model
    if (selectedCity && selectedModel) {
      loadAnalyticsData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCity, selectedModel, dateRange]);

  const handleCityChange = (city) => {
    setSelectedCityLocal(city);
    if (setSelectedCity) {
      setSelectedCity(city);
    }
  };

  const loadAnalyticsData = async () => {
    try {
      setLoading(true);
      setError(null); // Clear any previous errors
      
      // Use Promise.allSettled to handle errors gracefully
      const [realTimeResult, stationResult] = await Promise.allSettled([
        getRealTimeComparison(selectedCity),
        getStationPredictions(
          selectedCity, 
          new Date().toISOString().split('T')[0],
          selectedModel
        )
      ]);

      // Extract data from results, using empty objects on rejection
      const realTimeData = realTimeResult.status === 'fulfilled' 
        ? realTimeResult.value 
        : { predictions: [], pollutants: [] };
      
      const stationData = stationResult.status === 'fulfilled' 
        ? stationResult.value 
        : { predictions: [] };

      // Log warnings for rejected promises but don't treat as errors
      if (realTimeResult.status === 'rejected') {
        console.warn(`Real-time data not available for ${selectedCity}:`, realTimeResult.reason?.message);
      }
      if (stationResult.status === 'rejected') {
        console.warn(`Station predictions not available for ${selectedCity}:`, stationResult.reason?.message);
      }

      // Update analytics data
      setAnalyticsData({
        modelMetrics: processModelMetrics(modelPerformance),
        realTimeComparison: processRealTimeData(realTimeData),
        stationPredictions: processStationData(stationData),
        categoryDistribution: calculateAQIDistribution(realTimeData),
        pollutantTrends: calculatePollutantTrends(realTimeData)
      });
    } catch (err) {
      console.error('Failed to load analytics data:', err);
      // Only set error for unexpected errors, not for missing real-time data
      if (!err.message?.includes('real-time') && !err.message?.includes('404')) {
        setError(err.message || 'Failed to load data');
      }
      // Keep previous data on error
      setAnalyticsData(prev => ({
        ...prev
      }));
    } finally {
      setLoading(false);
    }
  };

  // Helpers
  const processModelMetrics = (performance = {}) => {
    return Object.entries(performance).map(([model, metrics]) => ({
      model,
      RMSE: Number(metrics?.RMSE || 0),
      MAE: Number(metrics?.MAE || 0),
      R2: Number(metrics?.R2 || 0) * 100
    }));
  };

  const processRealTimeData = (data = {}) => {
    // Handle real-time API response structure
    if (data.realtime && data.predictions) {
      // Real-time API returns: { realtime: { aqi, category }, predictions: { today: { predicted_aqi } }, accuracy: { today } }
      const realtime = data.realtime;
      const predictions = data.predictions;
      const accuracy = data.accuracy;
      
      const result = [];
      if (realtime.aqi && predictions.today) {
        result.push({
          timestamp: new Date().toLocaleString(),
          predicted: Number(predictions.today.predicted_aqi) || 0,
          actual: Number(realtime.aqi) || 0,
          difference: Math.abs((Number(predictions.today.predicted_aqi) || 0) - (Number(realtime.aqi) || 0)),
          accuracy: accuracy?.today || null
        });
      }
      return result;
    }
    
    // Handle array of predictions
    if (Array.isArray(data.predictions) && data.predictions.length > 0) {
      return data.predictions.map((p) => ({
        timestamp: new Date(p.timestamp || Date.now()).toLocaleString(),
        predicted: Number(p.predicted_aqi || p.predicted_AQI) || 0,
        actual: Number(p.actual_aqi || p.actual_AQI || p.aqi) || 0,
        difference: Math.abs((Number(p.predicted_aqi || p.predicted_AQI) || 0) - (Number(p.actual_aqi || p.actual_AQI || p.aqi) || 0)),
        accuracy: p.accuracy || null
      }));
    }
    
    return [];
  };

  const processStationData = (data = {}) => {
    if (!data?.predictions || !Array.isArray(data.predictions)) return [];
    return data.predictions
      .filter(s => s.prediction_type === 'station_level')
      .map((s) => ({
        name: s.station_name || s.station || 'unknown',
        prediction: Number(s.predicted_aqi) || 0,
        confidence: s.station_confidence === 'high' ? 0.9 : 0.7,
        lastUpdated: s.timestamp || ''
      }));
  };

  const calculateAQIDistribution = (data = {}) => {
    const categories = { Good: 0, Satisfactory: 0, Moderate: 0, Poor: 0, 'Very Poor': 0, Severe: 0 };
    if (!Array.isArray(data.predictions)) return Object.entries(categories).map(([name, value]) => ({ name, value }));
    data.predictions.forEach((pred) => {
      const aqi = Number(pred.actual_aqi) || 0;
      if (aqi <= 50) categories.Good++;
      else if (aqi <= 100) categories.Satisfactory++;
      else if (aqi <= 200) categories.Moderate++;
      else if (aqi <= 300) categories.Poor++;
      else if (aqi <= 400) categories['Very Poor']++;
      else categories.Severe++;
    });
    return Object.entries(categories).map(([name, value]) => ({ name, value }));
  };

  const calculatePollutantTrends = (data = {}) => {
    // Handle different response structures
    if (Array.isArray(data.pollutants) && data.pollutants.length > 0) {
      return data.pollutants.map((p) => ({
        name: p.name || p.pollutant || '',
        value: Number(p.concentration || p.value) || 0,
        limit: Number(p.limit || p.safe_limit) || 0,
        status: (Number(p.concentration || p.value) || 0) > (Number(p.limit || p.safe_limit) || 0) ? 'Above Limit' : 'Within Limit'
      }));
    }
    
    // Try to extract from realtime data if available
    if (data.realtime) {
      const pollutants = [];
      if (data.realtime.pm25 !== undefined) {
        pollutants.push({ name: 'PM2.5', value: data.realtime.pm25, limit: 60 });
      }
      if (data.realtime.pm10 !== undefined) {
        pollutants.push({ name: 'PM10', value: data.realtime.pm10, limit: 100 });
      }
      return pollutants;
    }
    
    return [];
  };

  if (loading && !analyticsData.modelMetrics.length) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold">Analytics Dashboard</h1>
              <p className="text-gray-600">Advanced AQI analysis & insights</p>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow p-12">
            <LoadingSpinner message="Loading analytics data..." size="lg" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold">Analytics Dashboard</h1>
            <p className="text-gray-600">Advanced AQI analysis & insights</p>
            {error && (
              <div className="mt-2 text-red-600 text-sm flex items-center">
                <AlertTriangle size={16} className="mr-1"/> {error}
              </div>
            )}
          </div>
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center">
              {systemStatus?.initialized && systemStatus?.status === 'running' ? (
                <span className="flex items-center text-green-600"><CheckCircle2 size={16} className="mr-1"/> System Active</span>
              ) : (
                <span className="flex items-center text-yellow-600"><XCircle size={16} className="mr-1"/> System {systemStatus?.status || 'Loading'}</span>
              )}
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-4 bg-white rounded-lg shadow p-4">
          <select 
            className="form-select rounded-md border-gray-300 focus:ring-2 focus:ring-blue-500" 
            value={selectedCity || 'Delhi'} 
            onChange={(e) => handleCityChange(e.target.value)}
            disabled={!Array.isArray(availableCities) || availableCities.length === 0}
          >
            {Array.isArray(availableCities) && availableCities.length > 0 ? (
              availableCities
                .filter(city => city && city !== 'Unknown')
                .map((city) => (
                  <option key={city} value={city}>{city}</option>
                ))
            ) : (
              <option value="">No cities available</option>
            )}
          </select>

          <select 
            className="form-select rounded-md border-gray-300 focus:ring-2 focus:ring-blue-500" 
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={!Array.isArray(availableModels) || availableModels.length === 0}
          >
            {Array.isArray(availableModels) && availableModels.length > 0 ? (
              availableModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))
            ) : (
              <option value="">No models available</option>
            )}
          </select>

          <select className="form-select rounded-md border-gray-300" value={dateRange} onChange={(e) => setDateRange(e.target.value)}>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex space-x-4 border-b border-gray-200 mb-4">
            {[{ id: 'overview', label: 'Overview', icon: BarChart2 }, { id: 'realtime', label: 'Real-time', icon: Activity }, { id: 'stations', label: 'Stations', icon: Map }, { id: 'models', label: 'Models', icon: TrendingUp }].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3 py-2 -mb-px ${activeTab === tab.id ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600 hover:text-gray-800'}`}
              >
                <tab.icon size={14} className="inline-block mr-2" />{tab.label}
              </button>
            ))}
          </div>

          {/* Content */}
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4 flex items-center"><TrendingUp className="mr-2" size={18}/>Model Performance</h2>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={analyticsData.modelMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="model" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="RMSE" fill="#3B82F6" name="RMSE" />
                      <Bar dataKey="MAE" fill="#10B981" name="MAE" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4 flex items-center"><PieIcon className="mr-2" size={18}/>AQI Distribution</h2>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={analyticsData.categoryDistribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                        {analyticsData.categoryDistribution.map((entry, idx) => (
                          <Cell key={`cell-${idx}`} fill={categoryColors[idx % categoryColors.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'realtime' && (
            <div className="grid grid-cols-1 gap-6">
              {analyticsData.realTimeComparison.length > 0 ? (
                <>
                  <div className="bg-white rounded-lg shadow p-6">
                    <h2 className="text-xl font-semibold mb-4 flex items-center"><Activity className="mr-2" size={18}/>Prediction vs Actual</h2>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={analyticsData.realTimeComparison}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="timestamp" />
                          <YAxis />
                          <Tooltip 
                            formatter={(value, name) => {
                              if (name === 'predicted') return [`Predicted: ${value.toFixed(1)}`, 'AQI'];
                              if (name === 'actual') return [`Actual: ${value.toFixed(1)}`, 'AQI'];
                              return [value, name];
                            }}
                          />
                          <Legend />
                          <Line type="monotone" dataKey="predicted" stroke="#3B82F6" name="Predicted AQI" strokeWidth={2} />
                          <Line type="monotone" dataKey="actual" stroke="#10B981" name="Actual AQI" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    {analyticsData.realTimeComparison[0]?.accuracy && (
                      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                        <p className="text-sm text-blue-800">
                          <strong>Accuracy:</strong> {analyticsData.realTimeComparison[0].accuracy.toFixed(1)}%
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="bg-white rounded-lg shadow p-6">
                    <h2 className="text-xl font-semibold mb-4 flex items-center"><AlertTriangle className="mr-2" size={18}/>Pollutant Levels</h2>
                    {analyticsData.pollutantTrends.length > 0 ? (
                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={analyticsData.pollutantTrends}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="value" fill="#3B82F6" name="Concentration" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="limit" fill="#EF4444" name="Safe Limit" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="h-80 flex items-center justify-center text-gray-500">
                        <div className="text-center">
                          <AlertTriangle size={48} className="mx-auto mb-2 text-gray-300" />
                          <p>No pollutant data available for {selectedCity}</p>
                          <p className="text-xs text-gray-400 mt-2">Pollutant data will appear here when available.</p>
                        </div>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="h-80 flex items-center justify-center text-gray-500">
                    <div className="text-center">
                      <Activity size={48} className="mx-auto mb-2 text-gray-300" />
                      <p className="text-lg font-medium mb-2">Real-time data not available</p>
                      <p className="text-sm">Real-time comparison data is not available for {selectedCity} at this time.</p>
                      <p className="text-xs text-gray-400 mt-2">This is expected for some cities. Try selecting a different city or check back later.</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'stations' && (
            <div className="grid grid-cols-1 gap-6">
              {analyticsData.stationPredictions.length > 0 ? (
                <>
                  <div className="bg-white rounded-lg shadow p-6">
                    <h2 className="text-xl font-semibold mb-4 flex items-center"><Map className="mr-2" size={18}/>Station Predictions</h2>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="prediction" name="Prediction" />
                          <YAxis dataKey="confidence" name="Confidence" />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                          <Scatter name="Stations" data={analyticsData.stationPredictions} fill="#3B82F6" />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="bg-white rounded-lg shadow p-6">
                    <h2 className="text-xl font-semibold mb-4">Station Details</h2>
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Station</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Prediction</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Last Updated</th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {analyticsData.stationPredictions.map((station, idx) => (
                            <tr key={idx}>
                              <td className="px-6 py-4 whitespace-nowrap">{station.name}</td>
                              <td className="px-6 py-4 whitespace-nowrap">{station.prediction}</td>
                              <td className="px-6 py-4 whitespace-nowrap">{(station.confidence * 100).toFixed(1)}%</td>
                              <td className="px-6 py-4 whitespace-nowrap">{station.lastUpdated ? new Date(station.lastUpdated).toLocaleString() : '—'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              ) : (
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="h-80 flex items-center justify-center text-gray-500">
                    <div className="text-center">
                      <Map size={48} className="mx-auto mb-2 text-gray-300" />
                      <p className="text-lg font-medium mb-2">No station data available</p>
                      <p className="text-sm">Station-level predictions are not available for {selectedCity} at this time.</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'models' && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Model Performance Metrics</h2>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">RMSE</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">MAE</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">R² Score</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {analyticsData.modelMetrics.map((m, idx) => (
                      <tr key={idx}>
                        <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">{m.model}</td>
                        <td className="px-6 py-4 whitespace-nowrap">{m.RMSE.toFixed(2)}</td>
                        <td className="px-6 py-4 whitespace-nowrap">{m.MAE.toFixed(2)}</td>
                        <td className="px-6 py-4 whitespace-nowrap">{m.R2.toFixed(2)}%</td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${m.model === selectedModel ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'}`}>
                            {m.model === selectedModel ? 'Active' : 'Available'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Analytics;
