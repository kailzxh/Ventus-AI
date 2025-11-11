import React from 'react';

const QuickStats = ({ cities = [] }) => {
  const count = Array.isArray(cities) ? cities.length : 0;
  const avg = Array.isArray(cities) && count > 0 ? (cities.reduce((s, c) => s + (c.aqi || 0), 0) / count).toFixed(1) : '—';
  return (
    <div className="grid grid-cols-3 gap-4">
      <div className="bg-white rounded-lg shadow p-4">Total Cities<div className="text-2xl font-bold">{count}</div></div>
      <div className="bg-white rounded-lg shadow p-4">Avg AQI<div className="text-2xl font-bold">{avg}</div></div>
      <div className="bg-white rounded-lg shadow p-4">Top City<div className="text-2xl font-bold">{cities[0]?.name || '—'}</div></div>
    </div>
  );
};

export default QuickStats;
