// frontend/src/pages/Dashboard.js
import React from 'react';
import { useAQI } from '../context/AQIContext';
import AQICard from '../components/AQICard';
import AQITrendChart from '../components/AQITrendChart';
import CityMap from '../components/CityMap';
import QuickStats from '../components/QuickStats';

const Dashboard = () => {
  const { currentAQI, loading } = useAQI();

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  const worstCities = [...currentAQI]
    .sort((a, b) => b.aqi - a.aqi)
    .slice(0, 6);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">AQI Dashboard</h1>
        <p className="text-gray-600">Real-time Air Quality Monitoring</p>
      </div>

      <QuickStats cities={currentAQI} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Worst Affected Cities</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {worstCities.map(city => (
                <AQICard key={city.city} city={city} />
              ))}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">AQI Trends</h2>
            <AQITrendChart cities={worstCities.slice(0, 3)} />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">City Map</h2>
          <CityMap cities={currentAQI} />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;