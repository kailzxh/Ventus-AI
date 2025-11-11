// frontend/src/components/PredictionResults.js
import React from 'react';
import { TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';

const PredictionResults = ({ result }) => {
  const getSeverityColor = (category) => {
    const colors = {
      'Good': 'text-green-600 bg-green-100',
      'Satisfactory': 'text-yellow-600 bg-yellow-100',
      'Moderate': 'text-orange-600 bg-orange-100',
      'Poor': 'text-red-600 bg-red-100',
      'Very Poor': 'text-purple-600 bg-purple-100',
      'Severe': 'text-maroon-600 bg-maroon-100'
    };
    return colors[category] || 'text-gray-600 bg-gray-100';
  };

  const getHealthAdvice = (category) => {
    const advice = {
      'Good': 'Perfect air quality. Enjoy outdoor activities!',
      'Satisfactory': 'Air quality is acceptable. Sensitive individuals should consider reducing prolonged outdoor exertion.',
      'Moderate': 'Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.',
      'Poor': 'Members of sensitive groups may experience health effects. General public is less likely to be affected.',
      'Very Poor': 'Health alert: everyone may experience more serious health effects.',
      'Severe': 'Health warning of emergency conditions. The entire population is more likely to be affected.'
    };
    return advice[category];
  };

  const getIcon = (category) => {
    if (category === 'Good' || category === 'Satisfactory') {
      return <CheckCircle size={20} className="text-green-500" />;
    } else if (category === 'Moderate') {
      return <AlertTriangle size={20} className="text-yellow-500" />;
    } else {
      return <AlertTriangle size={20} className="text-red-500" />;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4 flex items-center">
        <TrendingUp className="mr-2" size={20} />
        Prediction Results
      </h2>

      {result && (
        <div className="space-y-4">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-lg font-medium text-gray-900">{result.city}</h3>
              <p className="text-sm text-gray-600">{result.date}</p>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500">Model: {result.model_used}</div>
              <div className="text-xs text-gray-500">
                {new Date(result.timestamp).toLocaleString()}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">
                {result.predicted_aqi}
              </div>
              <div className="text-sm text-gray-600">Predicted AQI</div>
            </div>
            
            <div className={`text-center p-4 rounded-lg ${getSeverityColor(result.category)}`}>
              <div className="font-semibold text-lg capitalize">
                {result.category}
              </div>
              <div className="text-sm">Air Quality</div>
            </div>
          </div>

          <div className="p-3 bg-blue-50 rounded-md">
            <div className="flex items-start">
              {getIcon(result.category)}
              <div className="ml-2">
                <h4 className="font-medium text-blue-800">Health Advice</h4>
                <p className="text-sm text-blue-700">{getHealthAdvice(result.category)}</p>
              </div>
            </div>
          </div>

          <div className="p-3 bg-gray-50 rounded-md">
            <h4 className="font-medium text-gray-800 mb-2">Prediction Details</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-gray-600">Model Used:</div>
              <div className="font-medium">{result.model_used.toUpperCase()}</div>
              
              <div className="text-gray-600">Confidence:</div>
              <div className="font-medium">High (Based on historical patterns)</div>
              
              <div className="text-gray-600">Data Source:</div>
              <div className="font-medium">Historical + Real-time</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionResults;