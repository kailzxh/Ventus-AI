# backend/config.py
import os
from datetime import timedelta
from dotenv import load_dotenv  # <-- 1. IMPORT THIS

load_dotenv()  # <-- 2. ADD THIS LINE TO LOAD THE .ENV FILE

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
    # These will now correctly load from your .env file
    AQICN_API_KEY = os.environ.get('AQICN_API_KEY', 'demo')
    OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
    
    # Model Configuration
    PREDICTION_DAYS = 7
    MODEL_UPDATE_FREQUENCY = timedelta(hours=6)

# --- NEW AND UPDATED CONFIGURATIONS BELOW ---

REALTIME_UPDATE_INTERVAL = 300  # 5 minutes
MAX_API_RETRIES = 3
API_TIMEOUT = 10

# Indian cities coordinates (for fallback)
# *** UPDATED: Keys are now the canonical lowercase names ***
CITY_COORDINATES = {
    'delhi': (28.6139, 77.2090),
    'mumbai': (19.0760, 72.8777),
    'bengaluru': (12.9716, 77.5946),
    'chennai': (13.0827, 80.2707),
    'kolkata': (22.5726, 88.3639),
    'hyderabad': (17.3850, 78.4867),
    'ahmedabad': (23.0225, 72.5714),
    'pune': (18.5204, 73.8567),
}

# 1. *** NEW: THE SINGLE SOURCE OF TRUTH: CITY NORMALIZATION MAP ***
# All keys should be lowercase. All values are the canonical name.
CITY_NORMALIZATION_MAP = {
    # Delhi
    'delhi': 'delhi',
    'new delhi': 'delhi',
    'nct': 'delhi',
    'delhi nct': 'delhi',
    
    # Mumbai
    'mumbai': 'mumbai',
    'bombay': 'mumbai',
    'navi mumbai': 'mumbai',
    
    # Bengaluru
    'bangalore': 'bengaluru',
    'bengaluru': 'bengaluru',
    
    # Chennai
    'chennai': 'chennai',
    'madras': 'chennai',
    
    # Kolkata
    'kolkata': 'kolkata',
    'calcutta': 'kolkata',
    'howrah': 'kolkata', # Often considered part of Kolkata metro
    
    # Other Major Cities
    'hyderabad': 'hyderabad',
    'ahmedabad': 'ahmedabad',
    'pune': 'pune',
    'poona': 'pune',
    'surat': 'surat',
    'jaipur': 'jaipur',
    'lucknow': 'lucknow',
    'kanpur': 'kanpur',
    'cawnpore': 'kanpur',
    'nagpur': 'nagpur',
    'indore': 'indore',
    'thane': 'thane',
    'bhopal': 'bhopal',
    'visakhapatnam': 'visakhapatnam',
    'vizag': 'visakhapatnam',
    'patna': 'patna',
    'vadodara': 'vadodara',
    'baroda': 'vadodara',
    'ghaziabad': 'ghaziabad',
    'ludhiana': 'ludhiana',
    'agra': 'agra',
    'nashik': 'nashik',
    'nasik': 'nashik',
    'faridabad': 'faridabad',
    'meerut': 'meerut',
    'rajkot': 'rajkot',
    'kalyan-dombivli': 'kalyan-dombivli',
    'kalyan': 'kalyan-dombivli',
    'dombivli': 'kalyan-dombivli',
    'vasai-virar': 'vasai-virar',
    'vasai': 'vasai-virar',
    'virar': 'vasai-virar',
    'varanasi': 'varanasi',
    'benares': 'varanasi',
    'banaras': 'varanasi',
    'srinagar': 'srinagar',
    'aurangabad': 'aurangabad',
    'dhanbad': 'dhanbad',
    'amritsar': 'amritsar',
    'allahabad': 'prayagraj',
    'prayagraj': 'prayagraj',
    'ranchi': 'ranchi',
    'coimbatore': 'coimbatore',
    'jabalpur': 'jabalpur',
    'gwalior': 'gwalior',
    'vijayawada': 'vijayawada',
    'jodhpur': 'jodhpur',
    'madurai': 'madurai',
    'raipur': 'raipur',
    'kota': 'kota',
    'chandigarh': 'chandigarh'
}

# 2. *** NEW: LIST OF ALL VALID CANONICAL CITY NAMES FOR FILTERING ***
NORMALIZED_CITIES = sorted(list(set(CITY_NORMALIZATION_MAP.values())))

# 3. *** UPDATED: This list is now just for reference. The map is the primary source. ***
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