# backend/realtime_api.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
# --- FIX: Import the normalization map ---
from config import Config, INDIAN_CITIES, CITY_NORMALIZATION_MAP

class RealTimeAQI:
    """
    Real-time AQI data fetcher from multiple sources
    """
    
    def __init__(self):
        self.sources = {
            'waqi': {
                'name': 'World Air Quality Index',
                'base_url': 'http://api.waqi.info/feed/',
                'token': Config.AQICN_API_KEY
            },
            'openweather': {
                'name': 'OpenWeather Air Pollution',
                'base_url': 'http://api.openweathermap.org/data/2.5/air_pollution',
                'token': Config.OPENWEATHER_API_KEY
            }
        }
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes cache
        # --- FIX: Add the city map ---
        self.city_map = CITY_NORMALIZATION_MAP
    
    # --- FIX: Add a helper function ---
    def _get_canonical_city(self, city):
        """Helper to normalize city names."""
        # Get the canonical name, or default to the lowercased version
        return self.city_map.get(str(city).lower(), str(city).lower())

    def fetch_realtime_aqi(self, cities=None):
        """
        Fetch real-time AQI data for specified cities
        """
        if cities is None:
            cities = INDIAN_CITIES[:10]  # Limit to 10 cities for performance
        
        print(f"🌐 Fetching real-time AQI data for {len(cities)} cities...")
        realtime_data = []
        
        for city in cities:
            try:
                city_data = self._fetch_city_aqi(city)
                if city_data:
                    realtime_data.append(city_data)
                    print(f"✅ {city}: AQI {city_data['AQI']} ({self.get_aqi_category(city_data['AQI'])})")
                else:
                    print(f"❌ {city}: Failed to fetch data")
                    
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error fetching data for {city}: {e}")
                continue
        
        if realtime_data:
            df = pd.DataFrame(realtime_data)
            print(f"📊 Fetched real-time data for {len(df)} cities")
            return df
        else:
            print("⚠️  No real-time data fetched, using fallback data")
            return self._generate_fallback_data(cities)
    
    def _fetch_city_aqi(self, city):
        """
        Fetch AQI data for a single city from available sources
        """
        # Check cache first
        cache_key = f"{city}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_timeout:
                return cached_data
        
        # Try different data sources in order
        city_data = None
        
        # Source 1: World Air Quality Index (WAQI)
        if Config.AQICN_API_KEY and Config.AQICN_API_KEY != 'demo':
            city_data = self._fetch_from_waqi(city)
        
        # Source 2: OpenWeather (fallback)
        if not city_data and Config.OPENWEATHER_API_KEY:
            city_data = self._fetch_from_openweather(city)
        
        # Source 3: Generate realistic data (fallback)
        if not city_data:
            city_data = self._generate_realistic_data(city)
        
        # Cache the result
        if city_data:
            self.cache[cache_key] = (datetime.now(), city_data)
        
        return city_data
    
    def _fetch_from_waqi(self, city):
        """Fetch AQI data from World Air Quality Index API"""
        try:
            url = f"{self.sources['waqi']['base_url']}{city}/"
            params = {'token': self.sources['waqi']['token']}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'ok':
                    aqi_data = data['data']
                    
                    # --- FIX: Get the canonical name ---
                    canonical_city = self._get_canonical_city(city)
                    
                    return {
                        # --- FIX: Store the canonical name ---
                        'City': canonical_city,
                        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'AQI': aqi_data.get('aqi', 0),
                        'PM2.5': self._extract_pollutant_value(aqi_data, 'pm25'),
                        'PM10': self._extract_pollutant_value(aqi_data, 'pm10'),
                        'NO2': self._extract_pollutant_value(aqi_data, 'no2'),
                        'SO2': self._extract_pollutant_value(aqi_data, 'so2'),
                        'CO': self._extract_pollutant_value(aqi_data, 'co'),
                        'O3': self._extract_pollutant_value(aqi_data, 'o3'),
                        'Temperature': self._extract_pollutant_value(aqi_data, 't'),
                        'Humidity': self._extract_pollutant_value(aqi_data, 'h'),
                        'Pressure': self._extract_pollutant_value(aqi_data, 'p'),
                        'WindSpeed': self._extract_pollutant_value(aqi_data, 'w'),
                        'Source': 'waqi',
                        'Station': aqi_data.get('city', {}).get('name', city)
                    }
                    
        except Exception as e:
            print(f"WAQI API error for {city}: {e}")
            
        return None

    def _fetch_from_openweather(self, city):
        """Fetch air pollution data from OpenWeather API"""
        try:
            # First get coordinates for the city
            geo_url = "http://api.openweathermap.org/geo/1.0/direct"
            geo_params = {
                'q': f"{city},IN",
                'limit': 1,
                'appid': self.sources['openweather']['token']
            }
            
            geo_response = requests.get(geo_url, params=geo_params, timeout=10)
            
            if geo_response.status_code == 200:
                geo_data = geo_response.json()
                if geo_data:
                    lat = geo_data[0]['lat']
                    lon = geo_data[0]['lon']
                    
                    # Get air pollution data
                    pollution_url = self.sources['openweather']['base_url']
                    pollution_params = {
                        'lat': lat,
                        'lon': lon,
                        'appid': self.sources['openweather']['token']
                    }
                    
                    pollution_response = requests.get(pollution_url, params=pollution_params, timeout=10)
                    
                    if pollution_response.status_code == 200:
                        pollution_data = pollution_response.json()
                        
                        if 'list' in pollution_data and len(pollution_data['list']) > 0:
                            components = pollution_data['list'][0]['components']
                            main_aqi = pollution_data['list'][0]['main']['aqi']
                            
                            # Convert OpenWeather AQI to standard AQI
                            aqi_value = self._convert_openweather_aqi(main_aqi)
                            
                            # --- FIX: Get the canonical name ---
                            canonical_city = self._get_canonical_city(city)
                            
                            return {
                                # --- FIX: Store the canonical name ---
                                'City': canonical_city,
                                'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'AQI': aqi_value,
                                'PM2.5': components.get('pm2_5', 0),
                                'PM10': components.get('pm10', 0),
                                'NO2': components.get('no2', 0),
                                'SO2': components.get('so2', 0),
                                'CO': components.get('co', 0),
                                'O3': components.get('o3', 0),
                                'Source': 'openweather',
                                'Coordinates': f"{lat},{lon}"
                            }
                            
        except Exception as e:
            print(f"OpenWeather API error for {city}: {e}")
            
        return None

    def _generate_realistic_data(self, city):
        """Generate realistic AQI data when APIs are unavailable"""
        
        # --- FIX: Normalize the city *before* lookup ---
        canonical_city = self._get_canonical_city(city)

        # --- FIX: Keys must be lowercase to match canonical names ---
        base_aqi_levels = {
            'delhi': 280, 'mumbai': 160, 'bengaluru': 120, 'chennai': 140,
            'kolkata': 220, 'hyderabad': 130, 'ahmedabad': 180, 'pune': 110,
            'surat': 150, 'jaipur': 170, 'lucknow': 260, 'kanpur': 290,
            'nagpur': 140, 'indore': 130, 'thane': 150, 'bhopal': 120,
            'visakhapatnam': 110, 'patna': 240, 'vadodara': 160, 'ghaziabad': 270,
            'ludhiana': 190, 'agra': 200, 'nashik': 130, 'faridabad': 250
        }
        
        # Use the canonical city for the lookup
        base_aqi = base_aqi_levels.get(canonical_city, 150)
        
        # Add time-based variation
        hour = datetime.now().hour
        if 6 <= hour <= 10:  # Morning rush hour
            variation = np.random.normal(20, 5)
        elif 17 <= hour <= 20:  # Evening rush hour
            variation = np.random.normal(25, 5)
        else:
            variation = np.random.normal(0, 10)
        
        aqi = max(50, min(500, base_aqi + variation))
        
        return {
            # --- FIX: Store the canonical name ---
            'City': canonical_city,
            'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'AQI': int(aqi),
            'PM2.5': int(aqi * 0.48 + np.random.normal(0, 8)),
            'PM10': int(aqi * 0.77 + np.random.normal(0, 12)),
            'NO2': int(aqi * 0.28 + np.random.normal(0, 6)),
            'SO2': int(aqi * 0.22 + np.random.normal(0, 4)),
            'CO': int(aqi * 0.15 + np.random.normal(0, 2)),
            'O3': int(aqi * 0.10 + np.random.normal(0, 3)),
            'Temperature': np.random.normal(25, 5),
            'Humidity': np.random.normal(60, 15),
            'Source': 'generated',
            'Note': 'Simulated data - API unavailable'
        }

    def _generate_fallback_data(self, cities):
        """Generate fallback data when no APIs work"""
        fallback_data = []
        for city in cities:
            fallback_data.append(self._generate_realistic_data(city))
        return pd.DataFrame(fallback_data)

    def _extract_pollutant_value(self, aqi_data, pollutant):
        """Extract pollutant value from WAQI API response"""
        try:
            return aqi_data.get('iaqi', {}).get(pollutant, {}).get('v', 0)
        except:
            return 0

    def _convert_openweather_aqi(self, ow_aqi):
        """Convert OpenWeather AQI (1-5) to standard AQI scale"""
        conversion_map = {
            1: 50,   # Good
            2: 100,  # Fair
            3: 150,  # Moderate
            4: 200,  # Poor
            5: 300   # Very Poor
        }
        return conversion_map.get(ow_aqi, 150)

    def get_aqi_category(self, aqi):
        """Convert AQI value to category"""
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Satisfactory'
        elif aqi <= 200:
            return 'Moderate'
        elif aqi <= 300:
            return 'Poor'
        elif aqi <= 400:
            return 'Very Poor'
        else:
            return 'Severe'

    def get_health_advice(self, category):
        """Get health advice based on AQI category"""
        advice = {
            'Good': 'Air quality is satisfactory. Enjoy your normal outdoor activities.',
            'Satisfactory': 'Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.',
            'Moderate': 'Members of sensitive groups may experience health effects. General public is less likely to be affected.',
            'Poor': 'Everyone may begin to experience health effects. Members of sensitive groups may experience more serious health effects.',
            'Very Poor': 'Health alert: everyone may experience more serious health effects.',
            'Severe': 'Health warning of emergency conditions. The entire population is more likely to be affected.'
        }
        return advice.get(category, 'No data available.')

# Global instance
realtime_aqi = RealTimeAQI()

if __name__ == "__main__":
    # Test the real-time API
    print("🧪 Testing Real-time AQI API...")
    
    # Test with a few cities
    test_cities = ['Delhi', 'Mumbai', 'Bangalore']
    data = realtime_aqi.fetch_realtime_aqi(test_cities)
    
    print(f"\n📊 Test Results:")
    print(f"Fetched data for {len(data)} cities")
    
    for _, row in data.iterrows():
        category = realtime_aqi.get_aqi_category(row['AQI'])
        print(f"  {row['City']}: AQI {row['AQI']} ({category}) - Source: {row['Source']}")