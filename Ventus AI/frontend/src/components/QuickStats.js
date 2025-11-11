// frontend/src/components/QuickStats.js
import React from 'react';
import { MapPin, BarChart2, TrendingUp } from 'lucide-react';

const QuickStats = ({ cities = [] }) => {
  if (!Array.isArray(cities)) {
    cities = [];
  }

  const count = cities.length;
  const validCities = cities.filter(c => c && typeof c.aqi === 'number' && !isNaN(c.aqi));
  const avg = validCities.length > 0 
    ? (validCities.reduce((s, c) => s + (c.aqi || 0), 0) / validCities.length).toFixed(1)
    : '—';
  
  const worstCity = cities.length > 0 
    ? cities.reduce((worst, current) => {
        const worstAqi = worst?.aqi || 0;
        const currentAqi = current?.aqi || 0;
        return currentAqi > worstAqi ? current : worst;
      }, cities[0])
    : null;

  const stats = [
    {
      label: 'Total Cities',
      value: count,
      icon: MapPin,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50'
    },
    {
      label: 'Average AQI',
      value: avg,
      icon: BarChart2,
      color: 'text-green-600',
      bgColor: 'bg-green-50'
    },
    {
      label: 'Worst Affected',
      value: worstCity ? `${worstCity.city || worstCity.name || 'Unknown'}: ${worstCity.aqi || '—'}` : '—',
      icon: TrendingUp,
      color: 'text-red-600',
      bgColor: 'bg-red-50'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {stats.map((stat, index) => {
        const Icon = stat.icon;
        return (
          <div key={index} className="bg-white rounded-lg shadow p-6 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600 mb-1">{stat.label}</p>
                <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
              </div>
              <div className={`${stat.bgColor} rounded-full p-3`}>
                <Icon size={24} className={stat.color} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default QuickStats;
