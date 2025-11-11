// frontend/src/components/RealTimeCityCard.js
import React from 'react';
import { Activity, TrendingUp, TrendingDown, Minus, Calendar, AlertTriangle, CheckCircle, Info, Clock } from 'lucide-react';

const RealTimeCityCard = ({ realtimeData, cityName }) => {
  if (!realtimeData || !realtimeData.realtime) {
    return null;
  }

  const { realtime, predictions, accuracy, health_advice, trend_analysis } = realtimeData;

  const getAQIColor = (aqi) => {
    if (!aqi) return 'text-gray-500';
    if (aqi <= 50) return 'text-green-600';
    if (aqi <= 100) return 'text-yellow-600';
    if (aqi <= 200) return 'text-orange-600';
    if (aqi <= 300) return 'text-red-600';
    if (aqi <= 400) return 'text-purple-600';
    return 'text-red-700';
  };

  const getAQIBgColor = (aqi) => {
    if (!aqi) return 'bg-gray-100';
    if (aqi <= 50) return 'bg-green-50';
    if (aqi <= 100) return 'bg-yellow-50';
    if (aqi <= 200) return 'bg-orange-50';
    if (aqi <= 300) return 'bg-red-50';
    if (aqi <= 400) return 'bg-purple-50';
    return 'bg-red-100';
  };

  const getTrendIcon = (trend) => {
    if (trend === 'improving') return <TrendingDown className="text-green-500" size={20} />;
    if (trend === 'worsening') return <TrendingUp className="text-red-500" size={20} />;
    return <Minus className="text-gray-500" size={20} />;
  };

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    } catch {
      return dateString;
    }
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 90) return 'text-green-600';
    if (accuracy >= 70) return 'text-yellow-600';
    if (accuracy >= 50) return 'text-orange-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b pb-4">
        <div>
          <h3 className="text-2xl font-bold text-gray-900 capitalize">{cityName || realtimeData.city_requested || 'Unknown City'}</h3>
          <p className="text-sm text-gray-500 mt-1 flex items-center">
            <Clock size={14} className="mr-1" />
            Updated: {realtime.timestamp || 'Just now'}
          </p>
        </div>
        <div className={`px-4 py-2 rounded-lg ${getAQIBgColor(realtime.aqi)}`}>
          <div className="text-sm text-gray-600">Current AQI</div>
          <div className={`text-3xl font-bold ${getAQIColor(realtime.aqi)}`}>
            {Math.round(realtime.aqi)}
          </div>
          <div className="text-sm font-medium capitalize">{realtime.category}</div>
        </div>
      </div>

      {/* Real-time Data */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="text-sm text-gray-600 mb-1">PM2.5</div>
          <div className="text-2xl font-bold text-gray-900">{realtime.pm25 || '—'}</div>
          <div className="text-xs text-gray-500 mt-1">μg/m³</div>
        </div>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="text-sm text-gray-600 mb-1">PM10</div>
          <div className="text-2xl font-bold text-gray-900">{realtime.pm10 || '—'}</div>
          <div className="text-xs text-gray-500 mt-1">μg/m³</div>
        </div>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="text-sm text-gray-600 mb-1">Source</div>
          <div className="text-lg font-semibold text-gray-900 uppercase">{realtime.source || 'N/A'}</div>
        </div>
      </div>

      {/* Predictions */}
      {predictions && (
        <div className="space-y-4">
          <h4 className="text-lg font-semibold text-gray-900 flex items-center">
            <Calendar className="mr-2" size={20} />
            Predictions
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {predictions.today && (
              <div className="border rounded-lg p-4 bg-blue-50 border-blue-200">
                <div className="text-sm text-gray-600 mb-2">Today ({formatDate(predictions.today.date)})</div>
                <div className="text-2xl font-bold text-gray-900">{Math.round(predictions.today.predicted_aqi)}</div>
                <div className="text-sm font-medium capitalize text-gray-700">{predictions.today.category}</div>
                <div className="text-xs text-gray-500 mt-2">Model: {predictions.today.model_used || 'N/A'}</div>
                {accuracy && accuracy.today && (
                  <div className={`text-xs font-semibold mt-1 ${getAccuracyColor(accuracy.today)}`}>
                    Accuracy: {accuracy.today.toFixed(1)}%
                  </div>
                )}
              </div>
            )}

            {predictions.tomorrow && (
              <div className="border rounded-lg p-4 bg-yellow-50 border-yellow-200">
                <div className="text-sm text-gray-600 mb-2">Tomorrow ({formatDate(predictions.tomorrow.date)})</div>
                <div className="text-2xl font-bold text-gray-900">{Math.round(predictions.tomorrow.predicted_aqi)}</div>
                <div className="text-sm font-medium capitalize text-gray-700">{predictions.tomorrow.category}</div>
                <div className="text-xs text-gray-500 mt-2">Model: {predictions.tomorrow.model_used || 'N/A'}</div>
                <div className="text-xs text-gray-500 mt-1">Confidence: {predictions.tomorrow.confidence || 'N/A'}</div>
              </div>
            )}

            {predictions.next_week && (
              <div className="border rounded-lg p-4 bg-orange-50 border-orange-200">
                <div className="text-sm text-gray-600 mb-2">Next Week ({formatDate(predictions.next_week.date)})</div>
                <div className="text-2xl font-bold text-gray-900">{Math.round(predictions.next_week.predicted_aqi)}</div>
                <div className="text-sm font-medium capitalize text-gray-700">{predictions.next_week.category}</div>
                <div className="text-xs text-gray-500 mt-2">Model: {predictions.next_week.model_used || 'N/A'}</div>
                <div className="text-xs text-gray-500 mt-1">Confidence: {predictions.next_week.confidence || 'N/A'}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Accuracy Status */}
      {accuracy && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <CheckCircle className="text-blue-600 mr-2" size={20} />
              <span className="font-semibold text-blue-900">Prediction Accuracy</span>
            </div>
            <div className="text-right">
              <div className={`text-2xl font-bold ${getAccuracyColor(accuracy.today)}`}>
                {accuracy.today ? accuracy.today.toFixed(1) : '—'}%
              </div>
              <div className="text-sm text-blue-700">{accuracy.status || 'Good'}</div>
            </div>
          </div>
        </div>
      )}

      {/* Health Advice */}
      {health_advice && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-start">
            <Info className="text-yellow-600 mr-2 mt-0.5 flex-shrink-0" size={20} />
            <div>
              <h5 className="font-semibold text-yellow-900 mb-1">Health Advice</h5>
              <p className="text-sm text-yellow-800">{health_advice}</p>
            </div>
          </div>
        </div>
      )}

      {/* Trend Analysis */}
      {trend_analysis && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <h5 className="font-semibold text-gray-900 mb-3 flex items-center">
            <Activity className="mr-2" size={18} />
            Trend Analysis
          </h5>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-gray-600 mb-1">Short-term Trend</div>
              <div className="flex items-center space-x-2">
                {getTrendIcon(trend_analysis.short_term)}
                <span className="font-medium capitalize text-gray-900">{trend_analysis.short_term || 'Stable'}</span>
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600 mb-1">Long-term Trend</div>
              <div className="flex items-center space-x-2">
                {getTrendIcon(trend_analysis.long_term)}
                <span className="font-medium capitalize text-gray-900">{trend_analysis.long_term || 'Stable'}</span>
              </div>
            </div>
            {trend_analysis.confidence && (
              <div className="md:col-span-2">
                <div className="text-sm text-gray-600 mb-1">Confidence</div>
                <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-semibold rounded capitalize">
                  {trend_analysis.confidence}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default RealTimeCityCard;

