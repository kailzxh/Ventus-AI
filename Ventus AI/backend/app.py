# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import traceback

# Custom JSON encoder to handle numpy types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from config import Config
from data_loader import ComprehensiveAQIDataLoader
from preprocessor import AQIPreprocessor
from prediction import AQIPredictor
from realtime_api import RealTimeAQI  # Add realtime API import

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
app.json_encoder = CustomJSONEncoder
# Configure CORS properly
cors = CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://192.168.1.12:3000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
api = Api(app)

# Global variables
data_loader = ComprehensiveAQIDataLoader()
preprocessor = AQIPreprocessor()
predictor = AQIPredictor()
realtime_aqi = RealTimeAQI()  # Add realtime instance
historical_data = None
realtime_data = None
system_initialized = False

class HealthCheck(Resource):
    def get(self):
        try:
            models_loaded = False
            if hasattr(predictor, 'get_available_models'):
                available_models = predictor.get_available_models()
                models_loaded = len(available_models) > 0
            else:
                models_loaded = len(predictor.models) > 0 if hasattr(predictor, 'models') else False
            
            return {
                'status': 'healthy', 
                'timestamp': datetime.now().isoformat(),
                'system_initialized': system_initialized,
                'models_loaded': models_loaded
            }
        except Exception as e:
            return {
                'status': 'healthy', 
                'timestamp': datetime.now().isoformat(),
                'system_initialized': system_initialized,
                'models_loaded': False,
                'error': str(e)
            }
class CurrentAQI(Resource):
    def get(self):
        """Get current AQI for all cities with real-time data and model predictions"""
        try:
            global realtime_data, system_initialized, historical_data
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
            
            # Fetch real-time data
            print("🌐 Fetching real-time AQI data...")
            realtime_data = realtime_aqi.fetch_realtime_aqi()
            
            cities_data = []
            for _, row in realtime_data.iterrows():
                city_name = row['City']
                current_aqi = row['AQI']
                
                # Get model prediction for today
                today_prediction = predictor.predict_aqi(
                    city=city_name,
                    date=datetime.now().date(),
                    historical_data=historical_data,
                    model_type='nf_vae'  # Use NF-VAE for real-time comparison
                )
                
                # Get model prediction for tomorrow
                tomorrow_prediction = predictor.predict_aqi(
                    city=city_name,
                    date=datetime.now().date() + timedelta(days=1),
                    historical_data=historical_data,
                    model_type='nf_vae'
                )
                
                cities_data.append({
                    'city': city_name,
                    'current_aqi': current_aqi,
                    'current_category': predictor._get_aqi_category(current_aqi),
                    'predicted_today': today_prediction.get('predicted_aqi', 0) if today_prediction else 0,
                    'predicted_today_category': today_prediction.get('category', 'Unknown') if today_prediction else 'Unknown',
                    'predicted_tomorrow': tomorrow_prediction.get('predicted_aqi', 0) if tomorrow_prediction else 0,
                    'predicted_tomorrow_category': tomorrow_prediction.get('category', 'Unknown') if tomorrow_prediction else 'Unknown',
                    'accuracy': self._calculate_accuracy(current_aqi, today_prediction.get('predicted_aqi', 0) if today_prediction else 0),
                    'pm25': row.get('PM2.5', 0),
                    'pm10': row.get('PM10', 0),
                    'source': row.get('Source', 'unknown'),
                    'timestamp': row.get('Date', datetime.now().isoformat())
                })
            
            return _to_native({'cities': cities_data})
            
        except Exception as e:
            app.logger.error(f"Error in CurrentAQI: {str(e)}")
            return {'error': str(e)}, 500
    
    def _calculate_accuracy(self, actual, predicted):
        """Calculate prediction accuracy percentage"""
        if actual == 0 or predicted == 0:
            return 0
        error = abs(actual - predicted) / actual
        accuracy = max(0, 100 - (error * 100))
        return round(accuracy, 1)

class PredictAQI(Resource):
    def post(self):
        """Predict AQI for specific parameters"""
        try:
            global system_initialized, historical_data
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
                
            data = request.get_json()
            city = data.get('city', 'Delhi')
            date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
            model_type = data.get('model_type', 'auto')
            
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            prediction = predictor.predict_aqi(
                city=city,
                date=date,
                historical_data=historical_data,
                model_type=model_type
            )
            # If predictor returns a pandas Series accidentally, convert to dict
            if isinstance(prediction, pd.Series):
                prediction = prediction.to_dict()

            if prediction is not None:
                # Add real-time comparison if predicting for today
                if date == datetime.now().date():
                    realtime_city_data = self._get_realtime_city_data(city)
                    if realtime_city_data:
                        prediction['realtime_comparison'] = {
                            'realtime_aqi': realtime_city_data['AQI'],
                            'realtime_category': predictor._get_aqi_category(realtime_city_data['AQI']),
                            'accuracy': self._calculate_accuracy(
                                realtime_city_data['AQI'], 
                                prediction['predicted_aqi']
                            ),
                            'difference': round(prediction['predicted_aqi'] - realtime_city_data['AQI'], 2)
                        }
                return _to_native(prediction)
            else:
                return {'error': 'Prediction failed'}, 400
                
        except Exception as e:
            app.logger.error(f"Error in PredictAQI: {str(e)}")
            return {'error': str(e)}, 500
    
    def _get_realtime_city_data(self, city):
        """Get real-time data for a specific city"""
        global realtime_data
        try:
            if realtime_data is None:
                realtime_data = realtime_aqi.fetch_realtime_aqi([city])
            
            city_data = realtime_data[realtime_data['City'] == city]
            if len(city_data) > 0:
                # Return native dict instead of pandas Series
                return _to_native(city_data.iloc[0])
            return None
        except Exception as e:
            app.logger.error(f"Error getting realtime data for {city}: {e}")
            return None
    
    def _calculate_accuracy(self, actual, predicted):
        """Calculate prediction accuracy percentage"""
        if actual == 0 or predicted == 0:
            return 0
        error = abs(actual - predicted) / actual
        accuracy = max(0, 100 - (error * 100))
        return round(accuracy, 1)

class FuturePredictions(Resource):
    def get(self, city):
        """Get future AQI predictions for a city"""
        try:
            global system_initialized, historical_data
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
                
            days = request.args.get('days', 7, type=int)
            
            predictions = predictor.predict_future_aqi(
                city=city,
                days=days,
                historical_data=historical_data
            )
            
            return _to_native({'city': city, 'predictions': predictions})
            
        except Exception as e:
            app.logger.error(f"Error in FuturePredictions: {str(e)}")
            return {'error': str(e)}, 500

class CityList(Resource):
    def get(self):
        """Get list of available cities"""
        try:
            global historical_data, system_initialized
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
                
            if historical_data is not None and 'City' in historical_data.columns:
                cities = sorted(historical_data['City'].unique().tolist())
                return {'cities': cities, 'count': len(cities)}
            else:
                return {'cities': [], 'count': 0}
                
        except Exception as e:
            app.logger.error(f"Error in CityList: {str(e)}")
            return {'error': str(e)}, 500

class CityComparison(Resource):
    def get(self):
        """Compare AQI across all cities with real-time data and predictions"""
        try:
            global historical_data, system_initialized, realtime_data
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
                
            if historical_data is None:
                return {'error': 'No data available'}, 404
            
            # Fetch real-time data if not available
            if realtime_data is None:
                realtime_data = realtime_aqi.fetch_realtime_aqi()
            
            comparison_data = []
            for _, row in realtime_data.iterrows():
                city_name = row['City']
                current_aqi = row.get('AQI', 0)
                
                # Get prediction for tomorrow
                tomorrow_prediction = predictor.predict_aqi(
                    city=city_name,
                    date=datetime.now().date() + timedelta(days=1),
                    historical_data=historical_data
                )
                
                comparison_data.append({
                    'city': city_name,
                    'current_aqi': current_aqi,
                    'current_category': predictor._get_aqi_category(current_aqi),
                    'pm25': row.get('PM2.5', 0),
                    'prediction_tomorrow': tomorrow_prediction.get('predicted_aqi', 0) if tomorrow_prediction else 0,
                    'prediction_tomorrow_category': tomorrow_prediction.get('category', 'Unknown') if tomorrow_prediction else 'Unknown',
                    'trend': 'improving' if tomorrow_prediction and tomorrow_prediction.get('predicted_aqi', 0) < current_aqi else 'worsening',
                    'health_advice': realtime_aqi.get_health_advice(predictor._get_aqi_category(current_aqi))
                })
            
            # Sort by AQI (worst first)
            comparison_data.sort(key=lambda x: x['current_aqi'], reverse=True)
            
            return _to_native({'comparison': comparison_data})
            
        except Exception as e:
            app.logger.error(f"Error in CityComparison: {str(e)}")
            return {'error': str(e)}, 500

class ModelPerformance(Resource):
    def get(self):
        """Get model performance metrics"""
        try:
            global system_initialized
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
                
            # Load actual performance data from file
            performance_file = 'data/models/model_performance.json'
            nf_vae_performance_file = 'data/models/nf_vae_performance.json'
            
            performance_data = {}
            
            # Load baseline model performance
            if os.path.exists(performance_file):
                try:
                    with open(performance_file, 'r') as f:
                        baseline_performance = json.load(f)
                    performance_data.update(baseline_performance.get('models', {}))
                except Exception as e:
                    app.logger.warning(f"Could not load baseline performance: {e}")
            
            # Load NF-VAE performance
            if os.path.exists(nf_vae_performance_file):
                try:
                    with open(nf_vae_performance_file, 'r') as f:
                        nf_vae_performance = json.load(f)
                    performance_data['nf_vae'] = nf_vae_performance.get('nf_vae', {})
                except Exception as e:
                    app.logger.warning(f"Could not load NF-VAE performance: {e}")
            
            if performance_data:
                return {'model_performance': performance_data}
            else:
                # Fallback performance data
                return {'model_performance': {
                    'nf_vae': {'RMSE': 122.3, 'MAE': 94.2, 'R2': -0.038},
                    'random_forest': {'RMSE': 45.2, 'MAE': 32.1, 'R2': 0.782},
                    'gradient_boosting': {'RMSE': 48.7, 'MAE': 35.3, 'R2': 0.745}
                }}
                
        except Exception as e:
            app.logger.error(f"Error in ModelPerformance: {str(e)}")
            return {'error': str(e)}, 500

class AvailableModels(Resource):
    def get(self):
        """Get list of available prediction models"""
        try:
            global system_initialized
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
                
            models = []
            if hasattr(predictor, 'get_available_models'):
                models = predictor.get_available_models()
            elif hasattr(predictor, 'models'):
                models = list(predictor.models.keys())
            
            default_model = getattr(predictor, 'default_model', 'simple')
            
            return {
                'available_models': models,
                'default_model': default_model,
                'model_descriptions': {
                    'nf_vae': 'Normalizing Flow Variational Autoencoder (Primary)',
                    'random_forest': 'Random Forest Regressor',
                    'gradient_boosting': 'Gradient Boosting Regressor',
                    'simple': 'Simple Pattern-based Prediction'
                }
            }
            
        except Exception as e:
            app.logger.error(f"Error in AvailableModels: {str(e)}")
            return {'error': str(e)}, 500
class InitializeSystem(Resource):
    def post(self):
        """Manually initialize the system"""
        global system_initialized, historical_data, predictor, realtime_data
        try:
            print("🚀 Initializing AQI Prediction System...")
            
            # Load and preprocess historical data
            print("📥 Loading historical data...")
            historical_data = data_loader.load_historical_data()
            if historical_data is not None and len(historical_data) > 0:
                print(f"📊 Raw data loaded: {len(historical_data):,} records")
                historical_data = preprocessor.preprocess_data(historical_data, for_training=False)
                print(f"✅ Preprocessed data: {len(historical_data):,} records")
                
                if historical_data is not None and 'City' in historical_data.columns:
                    cities = historical_data['City'].unique()
                    print(f"🏙️  Cities available: {len(cities)}")
                else:
                    print("⚠️  No City column in processed data")
            else:
                print("❌ No historical data loaded")
                return {'status': 'error', 'message': 'No historical data available'}, 500
            
            # Load trained models
            print("🤖 Loading prediction models...")
            predictor.load_models()
            
            # Define available_models here before using it
            available_models = predictor.get_available_models() if hasattr(predictor, 'get_available_models') else []
            print(f"✅ Loaded {len(available_models)} models: {available_models}")
            
            # Fetch initial real-time data
            print("🌐 Fetching initial real-time data...")
            realtime_data = realtime_aqi.fetch_realtime_aqi()
            print(f"✅ Fetched real-time data for {len(realtime_data)} cities")
            
            system_initialized = True
            print("✅ System initialized successfully!")
            
            response_data = {
                'status': 'success', 
                'message': 'System initialized successfully',
                'data_loaded': historical_data is not None,
                'records_loaded': len(historical_data) if historical_data is not None else 0,
                'models_loaded': len(available_models),
                'cities_available': len(historical_data['City'].unique()) if historical_data is not None and 'City' in historical_data.columns else 0,
                'realtime_cities': len(realtime_data) if realtime_data is not None else 0,
                'available_models': available_models
            }
            
            return _to_native(response_data), 200
            
        except Exception as e:
            error_msg = f"System initialization failed: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            system_initialized = False
            return {'status': 'error', 'message': error_msg}, 500

class SystemStatus(Resource):
    def get(self):
        """Get detailed system status"""
        global system_initialized, historical_data, realtime_data
        
        try:
            available_models = []
            default_model = 'unknown'
            
            if hasattr(predictor, 'get_available_models'):
                available_models = predictor.get_available_models()
            elif hasattr(predictor, 'models'):
                available_models = list(predictor.models.keys())
            
            if hasattr(predictor, 'default_model'):
                default_model = predictor.default_model
            
            cities_available = 0
            if historical_data is not None and 'City' in historical_data.columns:
                cities_available = historical_data['City'].nunique()
            
            status = {
                'system_initialized': system_initialized,
                'timestamp': datetime.now().isoformat(),
                'data_status': {
                    'historical_data_loaded': historical_data is not None,
                    'historical_records': len(historical_data) if historical_data is not None else 0,
                    'cities_available': cities_available,
                    'realtime_data_loaded': realtime_data is not None,
                    'realtime_cities': len(realtime_data) if realtime_data is not None else 0
                },
                'model_status': {
                    'available_models': available_models,
                    'default_model': default_model
                }
            }
            
            return _to_native(status)
        except Exception as e:
            return {
                'system_initialized': system_initialized,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

class RealtimeComparison(Resource):
    def get(self, city):
        """Get detailed comparison between real-time data and model predictions"""
        try:
            global system_initialized, realtime_data, historical_data
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
            
            # Get real-time data for the city
            if realtime_data is None:
                realtime_data = realtime_aqi.fetch_realtime_aqi([city])
            
            city_realtime = realtime_data[realtime_data['City'] == city]
            if len(city_realtime) == 0:
                return {'error': f'No real-time data available for {city}'}, 404
            
            realtime_row = city_realtime.iloc[0]
            current_aqi = realtime_row['AQI']
            
            # Get predictions for different timeframes
            predictions = {
                'today': predictor.predict_aqi(city, datetime.now().date(), historical_data, 'nf_vae'),
                'tomorrow': predictor.predict_aqi(city, datetime.now().date() + timedelta(days=1), historical_data, 'nf_vae'),
                'next_week': predictor.predict_aqi(city, datetime.now().date() + timedelta(days=7), historical_data, 'nf_vae')
            }
            
            # Calculate accuracy for today's prediction
            today_accuracy = self._calculate_accuracy(current_aqi, predictions['today'].get('predicted_aqi', 0))
            
            comparison = {
                'city': city,
                'realtime': {
                    'aqi': current_aqi,
                    'category': predictor._get_aqi_category(current_aqi),
                    'pm25': realtime_row.get('PM2.5', 0),
                    'pm10': realtime_row.get('PM10', 0),
                    'timestamp': realtime_row.get('Date', datetime.now().isoformat()),
                    'source': realtime_row.get('Source', 'unknown')
                },
                'predictions': predictions,
                'accuracy': {
                    'today': today_accuracy,
                    'status': 'Excellent' if today_accuracy >= 90 else 'Good' if today_accuracy >= 80 else 'Fair' if today_accuracy >= 70 else 'Poor'
                },
                'health_advice': realtime_aqi.get_health_advice(predictor._get_aqi_category(current_aqi)),
                'trend_analysis': self._analyze_trend(current_aqi, predictions)
            }
            
            return _to_native(comparison)
            
        except Exception as e:
            app.logger.error(f"Error in RealtimeComparison: {str(e)}")
            return {'error': str(e)}, 500
    
    def _calculate_accuracy(self, actual, predicted):
        """Calculate prediction accuracy percentage"""
        if actual == 0 or predicted == 0:
            return 0
        error = abs(actual - predicted) / actual
        accuracy = max(0, 100 - (error * 100))
        return round(accuracy, 1)
    
    def _analyze_trend(self, current_aqi, predictions):
        """Analyze AQI trend based on predictions"""
        today_pred = predictions['today'].get('predicted_aqi', current_aqi)
        tomorrow_pred = predictions['tomorrow'].get('predicted_aqi', today_pred)
        next_week_pred = predictions['next_week'].get('predicted_aqi', tomorrow_pred)
        
        short_term_trend = 'improving' if tomorrow_pred < current_aqi else 'worsening'
        long_term_trend = 'improving' if next_week_pred < current_aqi else 'worsening'
        
        return {
            'short_term': short_term_trend,
            'long_term': long_term_trend,
            'confidence': 'high' if short_term_trend == long_term_trend else 'medium'
        }

# Register API routes
api.add_resource(HealthCheck, '/api/health')
api.add_resource(CurrentAQI, '/api/current-aqi')
api.add_resource(PredictAQI, '/api/predict')
api.add_resource(FuturePredictions, '/api/predict/<string:city>/future')
api.add_resource(CityList, '/api/cities')
api.add_resource(CityComparison, '/api/cities/comparison')
api.add_resource(ModelPerformance, '/api/models/performance')
api.add_resource(AvailableModels, '/api/models/available')
api.add_resource(InitializeSystem, '/api/initialize')
api.add_resource(SystemStatus, '/api/status')
api.add_resource(RealtimeComparison, '/api/realtime/<string:city>')  # New endpoint

@app.route('/')
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'AQI Prediction API',
        'status': 'running',
        'version': '2.0',
        'system_initialized': system_initialized,
        'endpoints': {
            '/api/health': 'Health check',
            '/api/status': 'Detailed system status',
            '/api/initialize': 'Initialize system (POST)',
            '/api/cities': 'Get available cities',
            '/api/cities/<city>/stations': 'Get stations for a city',
            '/api/current-aqi': 'Get current AQI for all cities',
            '/api/predict': 'Predict AQI for a city and date (POST)',
            '/api/predict/stations': 'Predict AQI for all stations in a city (POST)',
            '/api/predict/<city>/future': 'Predict AQI for next days',
            '/api/cities/comparison': 'Compare AQI across cities',
            '/api/models/available': 'Get available models',
            '/api/models/performance': 'Get model performance',
            '/api/realtime/<city>': 'Real-time vs prediction comparison'
        },
        'note': 'System must be initialized via /api/initialize before making predictions'
    })
   

def _to_native(obj):
    """Recursively convert numpy/pandas types to native Python types for JSON serialization."""
    # Basic types
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj

    # Numpy scalar types
    try:
        import numpy as _np
        if isinstance(obj, (_np.integer, _np.floating)):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # Pandas types
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Series):
            return _to_native(obj.to_dict())
        if isinstance(obj, _pd.DataFrame):
            return _to_native(obj.to_dict(orient='records'))
    except Exception:
        pass

    # Dicts and lists
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(v) for v in obj]

    # Datetime
    try:
        from datetime import datetime as _dt
        if isinstance(obj, _dt):
            return obj.isoformat()
    except Exception:
        pass

    # Fallback to string
    try:
        return str(obj)
    except Exception:
        return None


def initialize_system():
    """Initialize system on startup"""
    global system_initialized
    
    try:
        print("🔍 Checking for trained models...")
        
        # Check if any models exist
        model_files = [
            'data/models/best_nf_vae.pth',
            'data/models/random_forest_model.joblib', 
            'data/models/gradient_boosting_model.joblib'
        ]
        
        models_exist = any(os.path.exists(model_file) for model_file in model_files)
        
        if models_exist:
            print("🤖 Found trained models, attempting to initialize system...")
            
            # Use the InitializeSystem resource to initialize
            init_resource = InitializeSystem()
            result = init_resource.post()
            
            # Handle different return formats
            if isinstance(result, tuple) and len(result) == 2:
                # Format: (response_data, status_code)
                response_data, status_code = result
                if status_code == 200 and isinstance(response_data, dict) and response_data.get('status') == 'success':
                    system_initialized = True
                    print("✅ System auto-initialized successfully!")
                    print(f"📊 Loaded: {response_data.get('records_loaded', 0):,} records, {response_data.get('models_loaded', 0)} models")
                    print(f"🏙️  Cities: {response_data.get('cities_available', 0)} cities, {response_data.get('realtime_cities', 0)} real-time cities")
                else:
                    print(f"⚠️  System auto-initialization failed with status {status_code}")
                    print(f"💡 Response: {response_data}")
            elif isinstance(result, dict):
                # Format: just response_data (Flask-RESTful might wrap it)
                if result.get('status') == 'success':
                    system_initialized = True
                    print("✅ System auto-initialized successfully!")
                    print(f"📊 Loaded: {result.get('records_loaded', 0):,} records, {result.get('models_loaded', 0)} models")
                else:
                    print(f"⚠️  System auto-initialization returned: {result}")
            else:
                print(f"⚠️  Unexpected return type from initialization: {type(result)}")
                print(f"💡 Result: {result}")
                
        else:
            print("⚠️  No trained models found. Please train models first or initialize manually via /api/initialize")
            print("💡 To train models, run: python train_nf_vae.py && python train_models.py")
            
    except Exception as e:
        print(f"❌ Auto-initialization failed: {str(e)}")
        traceback.print_exc()
        print("💡 You can manually initialize the system via /api/initialize")
class StationPredictions(Resource):
    def post(self):
        """Predict AQI for all stations in a city"""
        try:
            global system_initialized, historical_data
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
                
            data = request.get_json()
            city = data.get('city', 'Delhi')
            date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
            model_type = data.get('model_type', 'auto')
            include_city_level = data.get('include_city_level', True)
            
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Get predictions for all stations
            station_predictions = predictor.predict_aqi_for_all_stations(
                city=city,
                date=date,
                historical_data=historical_data,
                model_type=model_type
            )
            
            # Add city-level prediction if requested
            if include_city_level:
                city_prediction = predictor.predict_aqi(
                    city=city,
                    date=date,
                    historical_data=historical_data,
                    model_type=model_type
                )
                # Mark city-level prediction
                city_prediction['prediction_type'] = 'city_level'
                city_prediction['station'] = 'City Average'
                station_predictions.append(city_prediction)
            
            return _to_native({
                'city': city,
                'date': date_str,
                'total_predictions': len(station_predictions),
                'station_predictions': len([p for p in station_predictions if p.get('prediction_type') == 'station_level']),
                'predictions': station_predictions,
                'summary': self._create_summary(station_predictions)
            })
                
        except Exception as e:
            app.logger.error(f"Error in StationPredictions: {str(e)}")
            return {'error': str(e)}, 500
    
    def _create_summary(self, predictions):
        """Create summary statistics for station predictions"""
        if not predictions:
            return {}
        
        aqi_values = [p['predicted_aqi'] for p in predictions]
        categories = [p['category'] for p in predictions]
        
        return {
            'average_aqi': round(np.mean(aqi_values), 2),
            'min_aqi': round(min(aqi_values), 2),
            'max_aqi': round(max(aqi_values), 2),
            'category_distribution': {cat: categories.count(cat) for cat in set(categories)},
            'worst_station': max(predictions, key=lambda x: x['predicted_aqi']),
            'best_station': min(predictions, key=lambda x: x['predicted_aqi'])
        }

class CityStations(Resource):
    def get(self, city):
        """Get all stations available for a city"""
        try:
            global historical_data, system_initialized
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
                
            if historical_data is None:
                return {'error': 'No historical data available'}, 404
            
            # Get all stations for the city
            stations = predictor._get_city_stations(city, historical_data)
            
            # Get station statistics
            station_stats = []
            for station_info in stations:
                station_name = station_info.get('station_name', 'Unknown')
                station_id = station_info.get('station_id', 'Unknown')
                
                # Try to find station data using multiple identifier columns
                station_data = None
                station_cols = ['Station', 'StationName', 'Site', 'Location', 'StationId']
                
                for col in station_cols:
                    if col in historical_data.columns:
                        station_mask = (historical_data['City'] == city) & (historical_data[col] == station_name)
                        if not station_mask.any():
                            station_mask = (historical_data['City'] == city) & (historical_data[col] == station_id)
                        
                        if station_mask.any():
                            station_data = historical_data[station_mask]
                            break
                
                if station_data is not None and len(station_data) > 0:
                    stats = {
                        'station_id': station_id,
                        'station_name': station_name,
                        'records_count': len(station_data),
                        'date_range': {
                            'start': station_data['Date'].min().strftime('%Y-%m-%d') if 'Date' in station_data.columns else 'Unknown',
                            'end': station_data['Date'].max().strftime('%Y-%m-%d') if 'Date' in station_data.columns else 'Unknown'
                        },
                        'avg_aqi': round(station_data['AQI'].mean(), 2) if 'AQI' in station_data.columns else 0,
                        'data_completeness': round((station_data.notna().sum() / len(station_data)).mean() * 100, 1)
                    }
                    station_stats.append(stats)
            
            return _to_native({
                'city': city,
                'total_stations': len(stations),
                'stations': stations,
                'station_statistics': station_stats
            })
            
        except Exception as e:
            app.logger.error(f"Error in CityStations: {str(e)}")
            return {'error': str(e)}, 500

# Register the new endpoints in your app.py (add these lines to the API routing section)
api.add_resource(StationPredictions, '/api/predict/stations')
api.add_resource(CityStations, '/api/cities/<string:city>/stations')
# Initialize system on startup
if __name__ == '__main__':
    import sys
    import logging
    
    # Configure logging to use sys.stderr for errors
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    logger = logging.getLogger('AQI-API')
    
    logger.info("Starting AQI Prediction API Server...")
    
    # Ensure data directories exist
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    # Check model files silently
    model_files = [
        'best_nf_vae.pth',
        'random_forest_model.joblib',
        'gradient_boosting_model.joblib',
        'nf_vae_performance.json',
        'model_performance.json'
    ]
    
    missing_models = []
    for model_file in model_files:
        full_path = f"data/models/{model_file}"
        if not os.path.exists(full_path):
            missing_models.append(model_file)
    
    if missing_models:
        logger.warning("Missing model files: %s", ", ".join(missing_models))
        logger.warning("Train models with: python train_nf_vae.py && python train_models.py")
    
    try:
        # Try to auto-initialize if models exist
        initialize_system()
    except Exception as e:
        logger.error("Auto-initialization failed: %s", str(e))
        logger.info("You can manually initialize via POST /api/initialize")
    
    # Run the application with minimal output
    logger.info("Server starting on http://0.0.0.0:5000")
    
    # Use werkzeug logging
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.ERROR)  # Only show errors
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent duplicate logging
    )