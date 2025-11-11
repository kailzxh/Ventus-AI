// frontend/src/components/AQITrendChart.js
import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import EmptyState from './EmptyState';
import { BarChart2 } from 'lucide-react';

const AQITrendChart = ({ cities = [] }) => {
  if (!Array.isArray(cities) || cities.length === 0) {
    return (
      <EmptyState
        icon={BarChart2}
        title="No trend data available"
        message="AQI trend data will appear here once city data is loaded."
      />
    );
  }

  // Normalize city data - handle both city.name and city.city properties
  const data = cities
    .filter(c => c && (c.aqi !== undefined && c.aqi !== null))
    .map((c, i) => ({
      name: c.city || c.name || `City ${i + 1}`,
      aqi: typeof c.aqi === 'number' ? Math.round(c.aqi) : 0,
      category: c.category || 'Unknown'
    }));

  if (data.length === 0) {
    return (
      <EmptyState
        icon={BarChart2}
        title="No valid AQI data"
        message="Unable to display trend chart. Please ensure cities have valid AQI values."
      />
    );
  }

  // Get color based on AQI value
  const getBarColor = (aqi) => {
    if (aqi <= 50) return '#10B981'; // Green
    if (aqi <= 100) return '#F59E0B'; // Yellow
    if (aqi <= 200) return '#F97316'; // Orange
    if (aqi <= 300) return '#EF4444'; // Red
    if (aqi <= 400) return '#8B5CF6'; // Purple
    return '#DC2626'; // Dark red
  };

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="name" 
            angle={-45}
            textAnchor="end"
            height={80}
            stroke="#6b7280"
            fontSize={12}
          />
          <YAxis stroke="#6b7280" fontSize={12} />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'white', 
              border: '1px solid #e5e7eb',
              borderRadius: '0.5rem',
              padding: '0.5rem'
            }}
            formatter={(value) => [`AQI: ${value}`, '']}
          />
          <Legend />
          <Bar 
            dataKey="aqi" 
            name="AQI"
            radius={[4, 4, 0, 0]}
            fill="#3B82F6"
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default AQITrendChart;
