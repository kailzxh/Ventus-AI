// frontend/src/components/ModelComparison.js
import React, { useState, useEffect } from 'react';
import { useAQI } from '../context/AQIContext';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import LoadingSpinner from './LoadingSpinner';

const ModelComparison = ({ city, date }) => {
  const { predictAQI, getAvailableModels } = useAQI();
  const [comparisonData, setComparisonData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);

  useEffect(() => {
    loadAvailableModels();
  }, []);

  useEffect(() => {
    if (city && date && availableModels.length > 0) {
      compareModels();
    }
  }, [city, date, availableModels]);

  const loadAvailableModels = async () => {
    try {
      const models = await getAvailableModels();
      setAvailableModels(models.available_models || []);
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const compareModels = async () => {
    if (!city || !date) return;

    setLoading(true);
    const results = [];

    try {
      // Test all available models
      for (const modelType of availableModels) {
        if (modelType === 'auto') continue; // Skip auto as it's not a specific model

        try {
          console.log(`🔮 Comparing model: ${modelType}`);
          const prediction = await predictAQI(city, date, modelType);
          
          results.push({
            model: modelType.toUpperCase(),
            predicted_aqi: Math.round(prediction.predicted_aqi),
            category: prediction.category,
            model_used: prediction.model_used,
            confidence: prediction.confidence,
            model_loaded: prediction.model_loaded,
            source: prediction.source
          });
        } catch (error) {
          console.error(`❌ Model ${modelType} failed:`, error);
          results.push({
            model: modelType.toUpperCase(),
            predicted_aqi: 0,
            category: 'Error',
            model_used: 'failed',
            error: error.message
          });
        }

        // Small delay to avoid overwhelming the API
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      setComparisonData(results);
    } catch (error) {
      console.error('Model comparison failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const chartData = comparisonData.map(item => ({
    model: item.model,
    AQI: item.predicted_aqi,
    category: item.category
  }));

  const getAQIColor = (aqi) => {
    if (aqi <= 50) return '#10B981';
    if (aqi <= 100) return '#F59E0B';
    if (aqi <= 200) return '#F97316';
    if (aqi <= 300) return '#EF4444';
    if (aqi <= 400) return '#8B5CF6';
    return '#DC2626';
  };

  if (!city || !date) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Comparison</h3>
        <p className="text-gray-500 text-center py-8">
          Select a city and date to compare model predictions
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Comparison</h3>
      
      {loading ? (
        <div className="flex justify-center py-8">
          <LoadingSpinner message="Comparing models..." />
        </div>
      ) : (
        <>
          {/* Chart */}
          <div className="h-64 mb-6">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" />
                <YAxis />
                <Tooltip 
                  formatter={(value, name) => [value, name === 'AQI' ? 'Predicted AQI' : name]}
                  labelFormatter={(label) => `Model: ${label}`}
                />
                <Legend />
                <Bar 
                  dataKey="AQI" 
                  fill="#3B82F6"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Results Table */}
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Model
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Predicted AQI
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Category
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Model Used
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {comparisonData.map((result, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                      {result.model}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                      <span className="font-bold">{result.predicted_aqi}</span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        result.category === 'Good' ? 'bg-green-100 text-green-800' :
                        result.category === 'Satisfactory' ? 'bg-yellow-100 text-yellow-800' :
                        result.category === 'Moderate' ? 'bg-orange-100 text-orange-800' :
                        result.category === 'Poor' ? 'bg-red-100 text-red-800' :
                        result.category === 'Very Poor' ? 'bg-purple-100 text-purple-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {result.category}
                      </span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                      {result.model_used}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm">
                      {result.error ? (
                        <span className="text-red-600 font-medium">Failed</span>
                      ) : result.model_loaded === false ? (
                        <span className="text-yellow-600 font-medium">Fallback</span>
                      ) : (
                        <span className="text-green-600 font-medium">Success</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Legend */}
          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-2 text-xs text-gray-600">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
              <span>Success: Model loaded and used</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
              <span>Fallback: Using simple model</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
              <span>Failed: Prediction error</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ModelComparison;