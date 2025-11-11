import React from 'react';

const CityMap = ({ cities = [] }) => {
  return (
    <div className="p-4">
      <p className="text-sm text-gray-500 mb-2">Map preview (stub)</p>
      <ul className="list-disc pl-5">
        {cities.slice(0, 10).map((c, i) => (
          <li key={i}>{c.city || c.name || `City ${i+1}`} — AQI: {c.aqi ?? '—'}</li>
        ))}
      </ul>
    </div>
  );
};

export default CityMap;
