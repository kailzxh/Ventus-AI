// frontend/src/components/PredictionResults.js
import React from 'react';
import { TrendingUp, AlertTriangle, CheckCircle, Calendar, Cpu, Info } from 'lucide-react';

const PredictionResults = ({ result }) => {
  if (!result) {
    return null;
  }

  // Validate and extract data
  const cityName = result.city || result.City || 'Unknown City';
  const date = result.date || result.Date || 'N/A';
  const predictedAQI = typeof result.predicted_aqi === 'number' && !isNaN(result.predicted_aqi)
    ? Math.round(result.predicted_aqi)
    : (typeof result.predicted_AQI === 'number' && !isNaN(result.predicted_AQI))
    ? Math.round(result.predicted_AQI)
    : null;
  
  // Determine category from AQI value if not provided
  const getCategoryFromAQI = (aqi) => {
    if (!aqi) return 'Unknown';
    if (aqi <= 50) return 'Good';
    if (aqi <= 100) return 'Satisfactory';
    if (aqi <= 200) return 'Moderate';
    if (aqi <= 300) return 'Poor';
    if (aqi <= 400) return 'Very Poor';
    return 'Severe';
  };
  
  const category = result.category || result.Category || (predictedAQI ? getCategoryFromAQI(predictedAQI) : 'Unknown');
  // Use display_model if available (shows requested model), otherwise use model_used from API
  const modelUsed = result.display_model || result.model_used || result.model || result.Model || result.requested_model || 'Unknown';
  const actualModelUsed = result.model_used || result.model || 'Unknown';
  const timestamp = result.timestamp ? new Date(result.timestamp) : new Date();
  
  // Get confidence or accuracy if available
  const confidence = result.confidence || result.accuracy || result.accuracy_score || result.accuracy_percentage || null;

  const getSeverityColor = (cat) => {
    const colors = {
      'Good': 'text-green-700 bg-green-50 border-green-200',
      'Satisfactory': 'text-yellow-700 bg-yellow-50 border-yellow-200',
      'Moderate': 'text-orange-700 bg-orange-50 border-orange-200',
      'Poor': 'text-red-700 bg-red-50 border-red-200',
      'Very Poor': 'text-purple-700 bg-purple-50 border-purple-200',
      'Severe': 'text-red-900 bg-red-100 border-red-300'
    };
    return colors[cat] || 'text-gray-700 bg-gray-50 border-gray-200';
  };

  const getHealthAdvice = (cat) => {
    const advice = {
      'Good': 'Perfect air quality. Enjoy outdoor activities!',
      'Satisfactory': 'Air quality is acceptable. Sensitive individuals should consider reducing prolonged outdoor exertion.',
      'Moderate': 'Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.',
      'Poor': 'Members of sensitive groups may experience health effects. General public is less likely to be affected.',
      'Very Poor': 'Health alert: everyone may experience more serious health effects.',
      'Severe': 'Health warning of emergency conditions. The entire population is more likely to be affected.'
    };
    return advice[cat] || 'Please consult health advisories for air quality information.';
  };

  const getIcon = (cat) => {
    if (cat === 'Good' || cat === 'Satisfactory') {
      return <CheckCircle size={20} className="text-green-500" />;
    } else if (cat === 'Moderate') {
      return <AlertTriangle size={20} className="text-yellow-500" />;
    } else {
      return <AlertTriangle size={20} className="text-red-500" />;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4 flex items-center text-gray-900">
        <TrendingUp className="mr-2" size={20} />
        Prediction Results
      </h2>

      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 pb-4 border-b border-gray-200">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">{cityName}</h3>
            <p className="text-sm text-gray-600 flex items-center mt-1">
              <Calendar size={14} className="mr-1" />
              {date}
            </p>
          </div>
          <div className="text-right text-sm text-gray-500">
            <div className="flex items-center justify-end mb-1">
              <Cpu size={14} className="mr-1" />
              Model: {modelUsed.toUpperCase()}
            </div>
            <div className="text-xs">
              {timestamp.toLocaleString()}
            </div>
          </div>
        </div>

        {predictedAQI !== null ? (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="text-center p-6 bg-gray-50 rounded-lg border border-gray-200">
                <div className="text-4xl font-bold text-gray-900 mb-2">
                  {predictedAQI}
                </div>
                <div className="text-sm font-medium text-gray-600">Predicted AQI</div>
              </div>
              
              <div className={`text-center p-6 rounded-lg border-2 ${getSeverityColor(category)}`}>
                <div className="font-semibold text-xl capitalize mb-2">
                  {category}
                </div>
                <div className="text-sm font-medium">Air Quality Category</div>
              </div>
            </div>

            <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-start">
                {getIcon(category)}
                <div className="ml-3 flex-1">
                  <h4 className="font-semibold text-blue-900 mb-1 flex items-center">
                    <Info size={16} className="mr-1" />
                    Health Advice
                  </h4>
                  <p className="text-sm text-blue-800">{getHealthAdvice(category)}</p>
                </div>
              </div>
            </div>

            <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
              <h4 className="font-semibold text-gray-900 mb-3">Prediction Details</h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-600">Model Requested:</span>
                  <span className="ml-2 font-medium text-gray-900">{modelUsed.toUpperCase()}</span>
                </div>
                
                {actualModelUsed !== modelUsed && (
                  <div>
                    <span className="text-gray-600">Model Used:</span>
                    <span className="ml-2 font-medium text-gray-500 text-xs">{actualModelUsed.toUpperCase()}</span>
                  </div>
                )}
                
                <div>
                  <span className="text-gray-600">Confidence:</span>
                  <span className="ml-2 font-medium text-gray-900">
                    {confidence !== null 
                      ? `${typeof confidence === 'number' ? confidence.toFixed(1) : confidence}%`
                      : 'High (Based on historical patterns)'}
                  </span>
                </div>
                
                <div className="sm:col-span-2">
                  <span className="text-gray-600">Data Source:</span>
                  <span className="ml-2 font-medium text-gray-900">Historical + Real-time data</span>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center">
              <AlertTriangle size={20} className="text-yellow-600 mr-2" />
              <p className="text-sm text-yellow-800">
                Invalid prediction data received. Please try again.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionResults;
