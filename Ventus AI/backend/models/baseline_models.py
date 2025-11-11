# backend/models/baseline_models.py
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

class BaselineModels:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'linear_regression': LinearRegression(),
        }
        self.trained_models = {}
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all baseline models"""
        print("🤖 Training Baseline Models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"   Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            results[name] = metrics
            
            print(f"   ✅ {name}: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}")
        
        return results
    
    def predict(self, model_name, X):
        """Make predictions using a specific model"""
        if model_name in self.trained_models:
            return self.trained_models[model_name].predict(X)
        else:
            raise ValueError(f"Model {model_name} not trained")
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred)
        }
    
    def save_models(self, path='data/models/'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.trained_models.items():
            filename = f"{path}/{name}_model.joblib"
            joblib.dump(model, filename)
            print(f"💾 Saved {name} to {filename}")
    
    def load_models(self, path='data/models/'):
        """Load trained models"""
        self.trained_models = {}
        for name in self.models.keys():
            filename = f"{path}/{name}_model.joblib"
            try:
                self.trained_models[name] = joblib.load(filename)
                print(f"📥 Loaded {name} from {filename}")
            except FileNotFoundError:
                print(f"❌ Model file not found: {filename}")