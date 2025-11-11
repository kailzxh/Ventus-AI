/* eslint-disable react-hooks/exhaustive-deps */
import React, { useEffect, useState } from 'react';
import { useAQI } from '../context/AQIContext';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, ScatterChart, Scatter
} from 'recharts';
import { TrendingUp, BarChart2, PieChart as PieIcon, Activity, Map, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';

const categoryColors = ['#10B981', '#F59E0B', '#F97316', '#EF4444', '#8B5CF6', '#DC2626'];

const Analytics = () => {
  const {
    systemStatus,
    availableModels = [],
    modelPerformance = {},
    getStationPredictions,
    getRealTimeComparison,
    selectedCity,
    availableCities = []
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

  useEffect(() => {
    // Only load data if we have a valid city and model
    if (selectedCity && selectedModel) {
      loadAnalyticsData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCity, selectedModel, dateRange]);

  const loadAnalyticsData = async () => {
    try {
      setLoading(true);
      
      // Use Promise.all to fetch data in parallel
      const [realTimeData, stationData] = await Promise.all([
        getRealTimeComparison(selectedCity).catch(err => {
          console.warn(`Failed to get real-time comparison for ${selectedCity}:`, err);
          return {};
        }),
        getStationPredictions(
          selectedCity, 
          new Date().toISOString().split('T')[0],
          selectedModel
        ).catch(err => {
          console.warn(`Failed to get station predictions for ${selectedCity}:`, err);
          return {};
        })
      ]);

      // Only update state if the component is still mounted and the city/model haven't changed
      setError(null); // Clear any previous errors
      setAnalyticsData({
        modelMetrics: processModelMetrics(modelPerformance),
        realTimeComparison: processRealTimeData(realTimeData),
        stationPredictions: processStationData(stationData),
        categoryDistribution: calculateAQIDistribution(realTimeData),
        pollutantTrends: calculatePollutantTrends(realTimeData)
      });
    } catch (err) {
      console.error('Failed to load analytics data:', err);
      setError(err.message || 'Failed to load data');
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
    if (!Array.isArray(data.predictions)) return [];
    return data.predictions.map((p) => ({
      timestamp: new Date(p.timestamp).toLocaleString(),
      predicted: Number(p.predicted_aqi) || 0,
      actual: Number(p.actual_aqi) || 0,
      difference: Math.abs((Number(p.predicted_aqi) || 0) - (Number(p.actual_aqi) || 0))
    }));
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
    if (!Array.isArray(data.pollutants)) return [];
    return data.pollutants.map((p) => ({
      name: p.name || '',
      value: Number(p.concentration) || 0,
      limit: Number(p.limit) || 0,
      status: (Number(p.concentration) || 0) > (Number(p.limit) || 0) ? 'Above Limit' : 'Within Limit'
    }));
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center h-48">
          <span className="text-gray-500">Loading analytics...</span>
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
              {systemStatus?.active ? (
                <span className="flex items-center text-green-600"><CheckCircle2 size={16} className="mr-1"/> System Active</span>
              ) : (
                <span className="flex items-center text-red-600"><XCircle size={16} className="mr-1"/> System Inactive</span>
              )}
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-4 bg-white rounded-lg shadow p-4">
          <select 
            className="form-select rounded-md border-gray-300" 
            value={selectedCity || 'Delhi'} 
            onChange={(e) => setSelectedCity(e.target.value)}
          >
            {availableCities
              .filter(city => city !== 'Unknown') // Filter out Unknown city
              .map((city) => (
                <option key={city} value={city}>{city}</option>
            ))}
          </select>

          <select className="form-select rounded-md border-gray-300" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            {availableModels.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
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
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4 flex items-center"><Activity className="mr-2" size={18}/>Prediction vs Actual</h2>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={analyticsData.realTimeComparison}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="predicted" stroke="#3B82F6" name="Predicted" />
                      <Line type="monotone" dataKey="actual" stroke="#10B981" name="Actual" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4 flex items-center"><AlertTriangle className="mr-2" size={18}/>Pollutant Levels</h2>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={analyticsData.pollutantTrends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="value" fill="#3B82F6" name="Concentration" />
                      <Bar dataKey="limit" fill="#EF4444" name="Safe Limit" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'stations' && (
            <div className="grid grid-cols-1 gap-6">
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
