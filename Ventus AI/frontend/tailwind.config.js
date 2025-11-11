// frontend/tailwind.config.js
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'aqi-good': '#10B981',
        'aqi-satisfactory': '#F59E0B',
        'aqi-moderate': '#F97316',
        'aqi-poor': '#EF4444',
        'aqi-very-poor': '#8B5CF6',
        'aqi-severe': '#DC2626',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}