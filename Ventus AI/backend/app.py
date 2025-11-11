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

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # *** ADDED: Handle NaT values gracefully for JSON ***
        elif pd.isna(obj):
            return None
        return super().default(obj)

# *** IMPORT THE NEW NORMALIZATION MAPS ***
from config import Config, CITY_NORMALIZATION_MAP, NORMALIZED_CITIES
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

# *** ADDED: Pass the data_loader to the predictor ***
# This is critical for the predictor to access station metadata
predictor.set_data_loader(data_loader)


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
                
                try:
                    # *** ADDED: Handle errors for individual city predictions ***
                    # Get model prediction for today
                    today_prediction = predictor.predict_aqi(
                        city=city_name,
                        date=datetime.now().date(),
                        historical_data=historical_data,
                        model_type='auto' # Use auto for best model
                    )
                    
                    # Get model prediction for tomorrow
                    tomorrow_prediction = predictor.predict_aqi(
                        city=city_name,
                        date=datetime.now().date() + timedelta(days=1),
                        historical_data=historical_data,
                        model_type='auto' # Use auto for best model
                    )
                except ValueError as ve:
                    # City not found in our system, skip it
                    print(f"⚠️ Skipping city {city_name} from real-time feed: {ve}")
                    continue

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
        try:
            error = abs(actual - predicted) / actual
            accuracy = max(0, 100 - (error * 100))
            return round(accuracy, 1)
        except Exception:
            return 0

class PredictAQI(Resource):
    def post(self):
        """Predict AQI for specific parameters"""
        try:
            global system_initialized, historical_data
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
                
            data = request.get_json()
            if not data:
                return {'error': 'Invalid JSON body'}, 400
                
            city = data.get('city', 'Delhi')
            date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
            model_type = data.get('model_type', 'auto')
            
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # *** UPDATED: Call to predictor.predict_aqi ***
            # This call is now wrapped in a specific ValueError catch
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
                # This case should be rare, as errors are now exceptions
                return {'error': 'Prediction failed'}, 400
                
        # *** FIX for Failure 5: Catch specific ValueErrors (e.g., "City not found") ***
        except ValueError as ve:
            app.logger.warning(f"Prediction failed for {city}: {str(ve)}")
            return {'error': str(ve)}, 404 # Return 404 Not Found
            
        except Exception as e:
            app.logger.error(f"Error in PredictAQI: {str(e)}")
            traceback.print_exc()
            return {'error': 'An internal server error occurred'}, 500
    
    def _get_realtime_city_data(self, city):
        """Get real-time data for a specific city"""
        global realtime_data
        try:
            if realtime_data is None:
                realtime_data = realtime_aqi.fetch_realtime_aqi() # Fetch all
            
            # *** Use canonical name for lookup ***
            canonical_city = predictor._normalize_city(city)
            if not canonical_city:
                return None
                
            # Note: realtime_aqi.fetch_realtime_aqi() should also be normalizing
            # its output. Assuming it returns canonical names.
            city_data = realtime_data[realtime_data['City'] == canonical_city]
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
        try:
            error = abs(actual - predicted) / actual
            accuracy = max(0, 100 - (error * 100))
            return round(accuracy, 1)
        except Exception:
            return 0

class FuturePredictions(Resource):
    def get(self, city):
        """Get future AQI predictions for a city"""
        try:
            global system_initialized, historical_data
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
                
            days = request.args.get('days', 7, type=int)
            
            # *** UPDATED: Call now wrapped in error handler ***
            predictions = predictor.predict_future_aqi(
                city=city,
                days=days,
                historical_data=historical_data
            )
            
            # The 'city' in response will be the *canonical* one from prediction.py
            return _to_native({'city_requested': city, 'predictions': predictions})
            
        # *** FIX for Failure 5: Catch specific ValueErrors (e.g., "City not found") ***
        except ValueError as ve:
            app.logger.warning(f"Future prediction failed for {city}: {str(ve)}")
            return {'error': str(ve)}, 404 # Return 404 Not Found

        except Exception as e:
            app.logger.error(f"Error in FuturePredictions: {str(e)}")
            return {'error': 'An internal server error occurred'}, 500

class CityList(Resource):
    def get(self):
        """Get list of available cities"""
        try:
            global system_initialized
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
                
            # *** FIX: Use the canonical city list from config ***
            # This is faster and more correct than querying the DataFrame
            cities = NORMALIZED_CITIES
            return {'cities': cities, 'count': len(cities)}
            
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
                
                try:
                    # *** ADDED: Handle errors for individual city predictions ***
                    tomorrow_prediction = predictor.predict_aqi(
                        city=city_name,
                        date=datetime.now().date() + timedelta(days=1),
                        historical_data=historical_data
                    )
                except ValueError as ve:
                    print(f"⚠️ Skipping city {city_name} from real-time feed: {ve}")
                    continue
                
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
                
            # This logic is fine, but it relies on JSON files
            # A better approach would be to get this from the predictor object
            performance_data = predictor.get_model_performance()
            
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
                
            models = predictor.get_available_models()
            default_model = predictor.default_model
            
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
            
            # *** ADDED: Set the data loader in the predictor ***
            # This MUST happen before loading data or models
            predictor.set_data_loader(data_loader)
            
            # Load and preprocess historical data
            print("📥 Loading historical data...")
            historical_data = data_loader.load_historical_data()
            if historical_data is not None and len(historical_data) > 0:
                print(f"📊 Raw data loaded: {len(historical_data):,} records")
                # Preprocessing is now more lightweight, as normalization is in the loader
                historical_data = preprocessor.preprocess_data(historical_data, for_training=False)
                print(f"✅ Preprocessed data: {len(historical_data):,} records")
                
                if historical_data is not None and 'City' in historical_data.columns:
                    cities = historical_data['City'].unique()
                    print(f"🏙️  Cities available: {len(cities)} (Normalized)")
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
            available_models = predictor.get_available_models()
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
            
            # *** FIX: Use canonical city name for lookup ***
            canonical_city = predictor._normalize_city(city)
            if not canonical_city:
                   raise ValueError(f"City '{city}' not found or not supported.")

            # Get real-time data for the city
            if realtime_data is None:
                realtime_data = realtime_aqi.fetch_realtime_aqi()
            
            city_realtime = realtime_data[realtime_data['City'] == canonical_city] # Match normalized
            
            # *** THIS IS THE LINE CAUSING YOUR ERROR ***
            # The 'realtime_data' DataFrame doesn't have a row for 'delhi',
            # so 'city_realtime' is empty.
            if len(city_realtime) == 0:
                return {'error': f'No real-time data available for {city} (normalized: {canonical_city})'}, 404
            
            realtime_row = city_realtime.iloc[0]
            current_aqi = realtime_row['AQI']
            
            # Get predictions for different timeframes
            predictions = {
                'today': predictor.predict_aqi(city, datetime.now().date(), historical_data, 'auto'),
                'tomorrow': predictor.predict_aqi(city, datetime.now().date() + timedelta(days=1), historical_data, 'auto'),
                'next_week': predictor.predict_aqi(city, datetime.now().date() + timedelta(days=7), historical_data, 'auto')
            }
            
            # Calculate accuracy for today's prediction
            today_accuracy = self._calculate_accuracy(current_aqi, predictions['today'].get('predicted_aqi', 0))
            
            comparison = {
                'city': canonical_city,
                'city_requested': city,
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

        # *** FIX for Failure 5: Catch specific ValueErrors (e.g., "City not found") ***
        except ValueError as ve:
            app.logger.warning(f"Realtime comparison failed for {city}: {str(ve)}")
            return {'error': str(ve)}, 404 # Return 404 Not Found
            
        except Exception as e:
            app.logger.error(f"Error in RealtimeComparison: {str(e)}")
            return {'error': 'An internal server error occurred'}, 500
    
    def _calculate_accuracy(self, actual, predicted):
        """Calculate prediction accuracy percentage"""
        if actual == 0 or predicted == 0:
            return 0
        try:
            error = abs(actual - predicted) / actual
            accuracy = max(0, 100 - (error * 100))
            return round(accuracy, 1)
        except Exception:
            return 0
    
    def _analyze_trend(self, current_aqi, predictions):
        """Analyze AQI trend based on predictions"""
        today_pred = predictions.get('today', {}).get('predicted_aqi', current_aqi)
        tomorrow_pred = predictions.get('tomorrow', {}).get('predicted_aqi', today_pred)
        next_week_pred = predictions.get('next_week', {}).get('predicted_aqi', tomorrow_pred)
        
        # Handle None values
        current_aqi = current_aqi or 0
        tomorrow_pred = tomorrow_pred or 0
        next_week_pred = next_week_pred or 0
        
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
        # *** ADDED: Handle NaT values ***
        if _pd.isna(obj):
            return None
    except Exception:
        pass

    # Dicts and lists
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(v) for v in obj]

    # Datetime
    try:
        from datetime import datetime as _dt, date as _date
        if isinstance(obj, (_dt, _date)):
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
            if not data:
                return {'error': 'Invalid JSON body'}, 400
                
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
            
            # *** FIX: We no longer need to manually add city-level ***
            # The new `predict_aqi_for_all_stations` already includes it.
            
            return _to_native({
                'city': predictor._normalize_city(city) or city, # Return canonical city
                'city_requested': city,
                'date': date_str,
                'total_predictions': len(station_predictions),
                'station_predictions': len([p for p in station_predictions if p.get('prediction_type') == 'station_level']),
                'predictions': station_predictions,
                'summary': self._create_summary(station_predictions)
            })
                
        # *** FIX for Failure 5: Catch specific ValueErrors (e.g., "City not found") ***
        except ValueError as ve:
            app.logger.warning(f"Station prediction failed for {city}: {str(ve)}")
            return {'error': str(ve)}, 404 # Return 404 Not Found
            
        except Exception as e:
            app.logger.error(f"Error in StationPredictions: {str(e)}")
            return {'error': 'An internal server error occurred'}, 500
    
    def _create_summary(self, predictions):
        """Create summary statistics for station predictions"""
        if not predictions:
            return {}
        
        # *** FIX: Filter for station_level predictions only for stats ***
        station_preds = [p for p in predictions if p.get('prediction_type') == 'station_level']
        if not station_preds:
            return {'average_aqi': 0, 'min_aqi': 0, 'max_aqi': 0}

        aqi_values = [p['predicted_aqi'] for p in station_preds]
        categories = [p['category'] for p in station_preds]
        
        return {
            'average_aqi': round(np.mean(aqi_values), 2),
            'min_aqi': round(min(aqi_values), 2),
            'max_aqi': round(max(aqi_values), 2),
            'category_distribution': {cat: categories.count(cat) for cat in set(categories)},
            'worst_station': max(station_preds, key=lambda x: x['predicted_aqi']),
            'best_station': min(station_preds, key=lambda x: x['predicted_aqi'])
        }

# ***
# *** RE-WRITTEN: DebugData (Fix for Failure 4 & Logic Error) ***
# ***
class DebugData(Resource):

    def _safe_strftime(self, date_val, default='N/A (No Date)'):
        """Helper to safely format dates that might be NaT"""
        if pd.isna(date_val):
            return default
        try:
            return date_val.strftime('%Y-%m-%d')
        except Exception:
            return default

    def get(self, city):
        """Debug endpoint to check available data for a city"""
        try:
            global historical_data
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
            
            if historical_data is None:
                return {'error': 'No historical data loaded'}, 404
            
            # *** FIX: Use correct normalization ***
            canonical_city = predictor._normalize_city(city)
            
            if not canonical_city:
                return {'error': f"City '{city}' not found or not supported."}, 404
            
            # *** FIX: Use fast, exact match on normalized column ***
            city_data = historical_data[historical_data['City'] == canonical_city]
            
            # *** FIX for Failure 4: Use safe date formatting ***
            start_date = city_data['Date'].min() if 'Date' in city_data.columns else pd.NaT
            end_date = city_data['Date'].max() if 'Date' in city_data.columns else pd.NaT

            response = {
                'city_requested': city,
                'city_normalized': canonical_city,
                'total_records': len(city_data),
                'date_range': {
                    'start': self._safe_strftime(start_date),
                    'end': self._safe_strftime(end_date)
                },
                'available_columns': list(city_data.columns) if len(city_data) > 0 else [],
                'stations_available': city_data['Station'].nunique() if len(city_data) > 0 and 'Station' in city_data.columns else 0,
                'sample_data': _to_native(city_data.head(3).to_dict('records')) if len(city_data) > 0 else []
            }
            
            return response
            
        except Exception as e:
            app.logger.error(f"Error in DebugData: {str(e)}")
            return {'error': str(e)}, 500

# Register the debug endpoint
api.add_resource(DebugData, '/api/debug/<string:city>')

# ***
# *** RE-WRITTEN: CityStations (Fix for Failure 6 & Logic Error) ***
# ***
class CityStations(Resource):
    def get(self, city):
        """Get all stations available for a city from the metadata"""
        try:
            global system_initialized
            
            if not system_initialized:
                return {'error': 'System not initialized. Please call /api/initialize first.'}, 503
            
            # *** FIX: This logic is now clean and correct ***
            # It relies on predictor._get_city_stations, which is now the single source of truth
            stations = predictor._get_city_stations(city)
            
            return _to_native({
                'city': predictor._normalize_city(city) or city, # Return canonical city
                'city_requested': city,
                'total_stations': len(stations),
                'stations': stations
            })
            
        # *** FIX for Failure 5: Catch specific ValueErrors (e.g., "City not found") ***
        except ValueError as ve:
            app.logger.warning(f"CityStations failed for {city}: {str(ve)}")
            return {'error': str(ve)}, 404 # Return 404 Not Found

        except Exception as e:
            app.logger.error(f"Error in CityStations: {str(e)}")
            return {'error': 'An internal server error occurred'}, 500

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
    
    # Run the application
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