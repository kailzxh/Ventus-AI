# backend/train_models.py
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from data_loader import ComprehensiveAQIDataLoader
from preprocessor import AQIPreprocessor

def train_simple_models():
    """Train simple models that work with the comprehensive data"""
    print("🚀 Starting Comprehensive Model Training...")
    
    # Load and preprocess data using comprehensive loader
    data_loader = ComprehensiveAQIDataLoader()
    preprocessor = AQIPreprocessor()
    
    print("📥 Loading comprehensive data...")
    df_raw = data_loader.load_historical_data()
    
    if df_raw is None or len(df_raw) == 0:
        print("❌ No raw data available for training")
        return None
    
    print(f"📊 Raw data: {len(df_raw):,} records")
    
    # Use simpler preprocessing for baseline models
    df_processed = preprocess_for_baseline_models(df_raw)
    
    if df_processed is None or len(df_processed) == 0:
        print("❌ No processed data available for training")
        return None
    
    print(f"📊 Processed data: {len(df_processed):,} records")
    
    # Get feature columns - use only reliable features
    feature_cols = get_reliable_features(df_processed)
    print(f"🎯 Using {len(feature_cols)} reliable features for training")
    print(f"📋 Features: {feature_cols}")
    
    # Prepare training data
    print("🔧 Preparing training data...")
    
    # Get features and target
    X = df_processed[feature_cols]
    y = df_processed['AQI']
    
    # Convert to numeric to handle any object types
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    # Handle missing values more carefully
    print("🔧 Handling missing values...")
    
    # Remove rows where target (AQI) is NaN
    valid_mask = ~pd.isna(y)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    print(f"📊 After removing NaN AQI: {len(X_clean):,} samples")
    
    # Fill remaining NaN in features with median
    X_filled = X_clean.copy()
    for col in X_filled.columns:
        if X_filled[col].isna().any():
            median_val = X_filled[col].median()
            X_filled[col] = X_filled[col].fillna(median_val)
            print(f"   🔧 Filled NaN in {col} with median: {median_val:.2f}")
    
    # Convert to numpy arrays
    X_values = X_filled.values
    y_values = y_clean.values
    
    print(f"📊 Final clean data: {X_values.shape[0]:,} samples, {X_values.shape[1]} features")
    
    if X_values.shape[0] == 0:
        print("❌ No valid data for training after preprocessing")
        return None
    
    # Split data - ensure we have enough samples
    test_size = min(0.2, 0.5 * X_values.shape[0] / X_values.shape[0])  # Adjust test size if needed
    
    if X_values.shape[0] < 10:
        print("⚠️  Very few samples available, using all for training")
        X_train, X_test, y_train, y_test = X_values, X_values, y_values, y_values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_values, y_values, test_size=test_size, random_state=42, shuffle=True
        )
    
    print(f"📊 Training data: {X_train.shape[0]:,} samples")
    print(f"📊 Test data: {X_test.shape[0]:,} samples")
    
    # Create models directory
    os.makedirs('data/models', exist_ok=True)
    
    # Train Simple Models with simpler configurations
    print("\n🤖 Training Simple Models...")
    
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=50,  # Reduced for stability
            random_state=42, 
            n_jobs=-1,
            max_depth=10,     # Reduced for stability
            min_samples_split=10,  # Increased for stability
            min_samples_leaf=5     # Increased for stability
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=50,  # Reduced for stability
            random_state=42,
            max_depth=6,      # Reduced for stability
            learning_rate=0.1,
            min_samples_split=10,  # Increased for stability
            min_samples_leaf=5     # Increased for stability
        ),
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Ensure predictions are within reasonable bounds
            y_pred = np.clip(y_pred, 0, 500)
            
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
            
            # Save feature information
            feature_info = {
                'feature_names': feature_cols,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {},
                'training_samples': len(X_train),
                'city_columns': [col for col in feature_cols if 'city' in col.lower()],
                'data_statistics': {
                    'X_train_shape': X_train.shape,
                    'y_train_stats': {
                        'mean': float(y_train.mean()),
                        'std': float(y_train.std()),
                        'min': float(y_train.min()),
                        'max': float(y_train.max())
                    }
                }
            }
            feature_filename = f"data/models/{name}_features.joblib"
            joblib.dump(feature_info, feature_filename)
            
            print(f"   ✅ {name}: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}")
            print(f"   💾 Saved to {filename}")
            
        except Exception as e:
            print(f"   ❌ Error training {name}: {e}")
            continue
    
    # Load NF-VAE performance if available
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
    
    # Save model performance
    performance_data = {
        'models': results,
        'training_info': {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(df_processed),
            'clean_data_size': X_values.shape[0],
            'features_used': len(feature_cols),
            'models_trained': list(trained_models.keys()),
            'nf_vae_available': os.path.exists('data/models/best_nf_vae.pth')
        },
        'feature_info': {
            'total_features': len(feature_cols),
            'feature_list': feature_cols,
            'data_quality': {
                'original_samples': len(df_raw),
                'processed_samples': len(df_processed),
                'training_samples': X_train.shape[0]
            }
        }
    }
    
    with open('data/models/model_performance.json', 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    print(f"\n✅ All models trained successfully!")
    
    # Show best model
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['RMSE'])
        print(f"📊 Best model: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.2f})")
        print(f"📊 Performance saved to data/models/model_performance.json")
    
    return performance_data

def preprocess_for_baseline_models(df):
    """Simplified preprocessing for baseline models"""
    print("🔧 Simplified preprocessing for baseline models...")
    
    if df is None or len(df) == 0:
        return None
    
    df_processed = df.copy()
    
    # 1. Basic cleaning
    if 'Date' in df_processed.columns:
        df_processed['Date'] = pd.to_datetime(df_processed['Date'], errors='coerce')
    
    # 2. Create basic temporal features (no complex encoding)
    if 'Date' in df_processed.columns:
        df_processed['Year'] = df_processed['Date'].dt.year
        df_processed['Month'] = df_processed['Date'].dt.month
        df_processed['DayOfWeek'] = df_processed['Date'].dt.dayofweek
        df_processed['IsWeekend'] = (df_processed['DayOfWeek'] >= 5).astype(int)
    
    # 3. Simple city encoding
    if 'City' in df_processed.columns:
        from sklearn.preprocessing import LabelEncoder
        city_encoder = LabelEncoder()
        df_processed['City_Encoded'] = city_encoder.fit_transform(df_processed['City'])
    
    # 4. Handle missing values in target
    if 'AQI' in df_processed.columns:
        initial_count = len(df_processed)
        df_processed = df_processed.dropna(subset=['AQI'])
        removed_count = initial_count - len(df_processed)
        if removed_count > 0:
            print(f"   🗑️  Removed {removed_count} rows with NaN AQI values")
    
    return df_processed

def get_reliable_features(df):
    """Get only reliable features that have sufficient data"""
    if df is None or len(df) == 0:
        return []
    
    # Define core features that are most important
    core_features = [
        # Primary pollutants
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
        # Basic temporal
        'Year', 'Month', 'DayOfWeek', 'IsWeekend',
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
    
    # Check which features actually exist and have data
    available_features = []
    
    for feature in core_features + additional_features:
        if feature in df.columns:
            # Check if feature has sufficient non-NaN values
            non_null_count = df[feature].notna().sum()
            if non_null_count > len(df) * 0.3:  # At least 30% non-null
                available_features.append(feature)
                print(f"   ✅ {feature}: {non_null_count:,} non-null values ({non_null_count/len(df)*100:.1f}%)")
            else:
                print(f"   ❌ {feature}: Only {non_null_count:,} non-null values ({non_null_count/len(df)*100:.1f}%) - skipping")
        else:
            print(f"   ❌ {feature}: Not in dataset")
    
    print(f"📊 Selected {len(available_features)} reliable features")
    
    return available_features

if __name__ == "__main__":
    train_simple_models()