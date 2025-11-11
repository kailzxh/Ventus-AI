# backend/train_models.py
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from data_loader import ComprehensiveAQIDataLoader
from preprocessor import AQIPreprocessor

def train_simple_models():
    """Train high-performance models using comprehensive data - OPTIMIZED VERSION"""
    print("🚀 Starting OPTIMIZED High-Performance Model Training...")
    
    # Load and preprocess data
    data_loader = ComprehensiveAQIDataLoader()
    df_raw = data_loader.load_historical_data()
    
    if df_raw is None or len(df_raw) == 0:
        print("❌ No raw data available for training")
        return None
    
    print(f"📊 Raw data: {len(df_raw):,} records")
    
    # --- 1. Preprocessing and Feature Engineering ---
    df_processed, city_encoder = preprocess_for_baseline_models(df_raw)
    
    if df_processed is None or len(df_processed) == 0:
        print("❌ No processed data available for training")
        return None
    
    print(f"📊 Processed data: {len(df_processed):,} records")

    # Save the city encoder
    os.makedirs('data/models', exist_ok=True)
    encoder_path = 'data/models/baseline_city_encoder.joblib'
    joblib.dump(city_encoder, encoder_path)
    print(f"💾 Saved City Encoder to {encoder_path}")
    
    # Get feature columns
    feature_cols = get_reliable_features(df_processed)
    print(f"🎯 Using {len(feature_cols)} reliable features for training")
    print(f"📋 Features: {feature_cols}")
    
    # --- 2. Data Splitting ---
    print("🔧 Preparing training data...")
    
    X = df_processed[feature_cols]
    y = df_processed['AQI']
    
    # Use smaller test size for faster training
    test_size = min(0.1, 5000 / len(X))  # Use 10% or max 5k samples
    
    if len(X) < 10:
        print("❌ Not enough data to train.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    
    print(f"📊 Training data: {X_train.shape[0]:,} samples")
    print(f"📊 Test data: {X_test.shape[0]:,} samples")
    
    # --- 3. Simple Imputation (FASTER) ---
    print("🔧 Applying simple imputation (median)...")
    
    # Use simple median imputation instead of IterativeImputer
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)
    
    # Save the imputer
    imputer_path = 'data/models/baseline_imputer.joblib'
    joblib.dump(imputer, imputer_path)
    print(f"💾 Saved Imputer to {imputer_path}")

    # --- 4. OPTIMIZED Model Training ---
    print("\n🤖 Training OPTIMIZED Models...")
    
    # Use smaller, faster models with fewer hyperparameters
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=20, 
            random_state=42, 
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'xgboost': XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        ),
        'lightgbm': LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"--- Training {name} ---")
        try:
            # DIRECT TRAINING - No hyperparameter search for speed
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            y_pred = np.clip(y_pred, 0, 600)  # Clip predictions
            
            # Calculate metrics
            metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred)
            }
            results[name] = metrics
            
            # Save model
            filename = f"data/models/{name}_model.joblib"
            joblib.dump(model, filename)
            
            # Save feature info
            feature_info = {
                'feature_names': feature_cols,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {},
                'training_samples': len(X_train),
                'imputer_used': 'SimpleImputer(median)',
                'model_params': model.get_params()
            }
            feature_filename = f"data/models/{name}_features.joblib"
            joblib.dump(feature_info, feature_filename)
            
            print(f"   ✅ {name}: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}")
            print(f"   💾 Saved to {filename}")

        except Exception as e:
            print(f"   ❌ Error training {name}: {e}")
            continue
    
    # --- 5. Load NF-VAE performance if available ---
    nf_vae_performance_path = 'data/models/nf_vae_performance.json'
    if os.path.exists(nf_vae_performance_path):
        try:
            with open(nf_vae_performance_path, 'r') as f:
                nf_vae_perf = json.load(f)
            
            nf_vae_metrics = nf_vae_perf.get('nf_vae', {}).get('aqi_only', {})
            if nf_vae_metrics:
                results['nf_vae'] = {
                    'RMSE': nf_vae_metrics.get('RMSE', 0),
                    'MAE': nf_vae_metrics.get('MAE', 0),
                    'R2': nf_vae_metrics.get('R2', 0),
                    'MSE': nf_vae_metrics.get('RMSE', 0) ** 2
                }
                print(f"   ✅ NF-VAE: RMSE={nf_vae_metrics.get('RMSE', 0):.2f}, R2={nf_vae_metrics.get('R2', 0):.3f}")
            else:
                print("   ⚠️  NF-VAE performance metrics not found")
        except Exception as e:
            print(f"   ❌ Error loading NF-VAE performance: {e}")
    
    # --- 6. Save performance results ---
    performance_data = {
        'models': results,
        'training_info': {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(df_processed),
            'clean_data_size': len(X),
            'features_used': len(feature_cols),
            'models_trained': list(trained_models.keys()),
            'training_strategy': 'OPTIMIZED_FAST_TRAINING'
        },
        'feature_info': {
            'total_features': len(feature_cols),
            'feature_list': feature_cols,
        }
    }
    
    with open('data/models/model_performance.json', 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    print(f"\n✅ All models trained successfully!")
    
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['RMSE'])
        print(f"🏆 Best model: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.2f})")
        print(f"📊 Performance saved to data/models/model_performance.json")
    
    return performance_data

def preprocess_for_baseline_models(df):
    """Simplified preprocessing for baseline models"""
    print("🔧 Advanced preprocessing for baseline models...")
    
    if df is None or len(df) == 0:
        return None, None
    
    df_processed = df.copy()
    
    # 1. Basic cleaning
    if 'Date' in df_processed.columns:
        df_processed['Date'] = pd.to_datetime(df_processed['Date'], errors='coerce')
    
    # 2. Create temporal features
    if 'Date' in df_processed.columns:
        valid_dates = df_processed['Date'].notna()
        df_processed['Year'] = df_processed['Date'].dt.year
        df_processed['DayOfYear'] = df_processed['Date'].dt.dayofyear
        df_processed['Month'] = df_processed['Date'].dt.month
        df_processed['DayOfWeek'] = df_processed['Date'].dt.dayofweek
        
        # Cyclical Features
        df_processed['Month_sin'] = np.sin(2 * np.pi * df_processed['Month'] / 12)
        df_processed['Month_cos'] = np.cos(2 * np.pi * df_processed['Month'] / 12)
        df_processed['DayOfWeek_sin'] = np.sin(2 * np.pi * df_processed['DayOfWeek'] / 7)
        df_processed['DayOfWeek_cos'] = np.cos(2 * np.pi * df_processed['DayOfWeek'] / 7)
        df_processed['DayOfYear_sin'] = np.sin(2 * np.pi * df_processed['DayOfYear'] / 365.25)
        df_processed['DayOfYear_cos'] = np.cos(2 * np.pi * df_processed['DayOfYear'] / 365.25)
        
        # Fill NaNs
        temporal_cols = ['Year', 'Month', 'DayOfWeek', 'DayOfYear', 'Month_sin', 'Month_cos', 
                         'DayOfWeek_sin', 'DayOfWeek_cos', 'DayOfYear_sin', 'DayOfYear_cos']
        for col in temporal_cols:
            if col.startswith('Year'):
                df_processed[col] = df_processed[col].fillna(2020)
            else:
                df_processed[col] = df_processed[col].fillna(0)
    
    # 3. Simple city encoding
    city_encoder = LabelEncoder()
    if 'City' in df_processed.columns:
        df_processed['City'] = df_processed['City'].fillna('Unknown')
        df_processed['City_Encoded'] = city_encoder.fit_transform(df_processed['City'])
        print(f"   ...City Encoder fit on {len(city_encoder.classes_)} cities.")
    else:
        df_processed['City_Encoded'] = 0
        city_encoder.fit(['Unknown'])
        print("   ...No 'City' column found. Using dummy 'City_Encoded'.")

    # 4. Handle missing values in target
    if 'AQI' in df_processed.columns:
        initial_count = len(df_processed)
        df_processed = df_processed.dropna(subset=['AQI'])
        removed_count = initial_count - len(df_processed)
        if removed_count > 0:
            print(f"   🗑️  Removed {removed_count} rows with NaN AQI values")
    
    return df_processed, city_encoder

def get_reliable_features(df):
    """Get only reliable features that have sufficient data"""
    if df is None or len(df) == 0:
        return []
    
    print("🔎 Checking for reliable features...")
    
    # Define core features
    core_features = [
        # Primary pollutants
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
        # Basic temporal
        'Year',
        # Cyclical temporal
        'Month_sin', 'Month_cos', 
        'DayOfWeek_sin', 'DayOfWeek_cos',
        'DayOfYear_sin', 'DayOfYear_cos',
        # City encoding
        'City_Encoded'
    ]
    
    # Additional features that might be available
    additional_features = [
        # Meteorological
        'Temperature', 'Humidity', 'Pressure', 'Wind_Speed',
        # Secondary pollutants
        'NO', 'NOx', 'NH3', 'Benzene', 'Toluene', 'Xylene'
    ]
    
    available_features = []
    
    for feature in core_features + additional_features:
        if feature in df.columns:
            non_null_count = df[feature].notna().sum()
            # At least 30% non-null
            if non_null_count > len(df) * 0.3: 
                available_features.append(feature)
                print(f"   ✅ {feature}: {non_null_count:,} non-null values ({non_null_count/len(df)*100:.1f}%)")
            else:
                print(f"   ❌ {feature}: Only {non_null_count:,} non-null values ({non_null_count/len(df)*100:.1f}%) - skipping")
    
    print(f"📊 Selected {len(available_features)} reliable features")
    
    return available_features

if __name__ == "__main__":
    train_simple_models()