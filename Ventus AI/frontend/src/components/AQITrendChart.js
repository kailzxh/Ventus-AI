import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const AQITrendChart = ({ cities = [] }) => {
  // cities is expected to be an array of objects { name, aqi }
  const data = Array.isArray(cities) ? cities.map((c, i) => ({ name: c.name || `City ${i+1}`, aqi: c.aqi || 0 })) : [];
  return (
    <div style={{ width: '100%', height: 200 }} className="bg-white rounded-lg p-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="aqi" fill="#3B82F6" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default AQITrendChart;
