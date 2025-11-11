# backend/config.py
import os
from datetime import timedelta

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'aqi-prediction-secret-key')
    DEBUG = os.environ.get('DEBUG', False)
    
    # CORS Configuration
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # API Configuration
    API_PREFIX = '/api/v1'
    
    # Data Configuration
    DATA_PATH = 'data/'
    MODEL_PATH = 'data/models/'
    
    # Real-time API Configuration
    AQICN_API_KEY = os.environ.get('AQICN_API_KEY', 'demo')
    OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
    
    # Model Configuration
    PREDICTION_DAYS = 7
    MODEL_UPDATE_FREQUENCY = timedelta(hours=6)

# Add this to your existing config.py
REALTIME_UPDATE_INTERVAL = 300  # 5 minutes
MAX_API_RETRIES = 3
API_TIMEOUT = 10

# Indian cities coordinates (for fallback)
CITY_COORDINATES = {
    'Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777),
    'Bangalore': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707),
    'Kolkata': (22.5726, 88.3639),
    'Hyderabad': (17.3850, 78.4867)
}

# Indian cities for real-time data collection
# In config.py - expand the INDIAN_CITIES list
INDIAN_CITIES = [
    'Delhi', 'New Delhi', 'NCT', 'Delhi NCT',
    'Mumbai', 'Bombay', 
    'Bangalore', 'Bengaluru',
    'Chennai', 'Madras',
    'Kolkata', 'Calcutta',
    'Hyderabad', 
    'Ahmedabad',
    'Pune', 'Poona',
    'Surat',
    'Jaipur',
    'Lucknow',
    'Kanpur', 'Cawnpore',
    'Nagpur',
    'Indore',
    'Thane',
    'Bhopal',
    'Visakhapatnam', 'Vizag', 'Visakhapatnam',
    'Patna',
    'Vadodara', 'Baroda',
    'Ghaziabad',
    'Ludhiana',
    'Agra',
    'Nashik', 'Nasik',
    'Faridabad',
    'Meerut',
    'Rajkot',
    'Kalyan-Dombivli', 'Kalyan', 'Dombivli',
    'Vasai-Virar', 'Vasai', 'Virar',
    'Varanasi', 'Benares', 'Banaras',
    'Srinagar',
    'Aurangabad',
    'Dhanbad',
    'Amritsar',
    'Navi Mumbai',
    'Allahabad', 'Prayagraj',
    'Ranchi',
    'Howrah',
    'Coimbatore',
    'Jabalpur',
    'Gwalior',
    'Vijayawada',
    'Jodhpur',
    'Madurai',
    'Raipur',
    'Kota',
    'Chandigarh'
]


