# backend/train_nf_vae.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.nf_vae import SequentialVAE, GRU_VAE_Trainer
from data_loader import ComprehensiveAQIDataLoader
from preprocessor import AQIPreprocessor

def enhance_aqi_features(df):
    """Add AQI-specific features for better prediction"""
    df = df.copy()
    
    # AQI category features (if AQI available)
    if 'AQI' in df.columns:
        df['AQI_Good'] = ((df['AQI'] <= 50) & (df['AQI'] > 0)).astype(float)
        df['AQI_Moderate'] = ((df['AQI'] > 50) & (df['AQI'] <= 100)).astype(float)
        df['AQI_Unhealthy_Sensitive'] = ((df['AQI'] > 100) & (df['AQI'] <= 150)).astype(float)
        df['AQI_Unhealthy'] = (df['AQI'] > 150).astype(float)
    
    # Pollutant ratios that correlate with AQI
    if all(col in df.columns for col in ['PM2.5', 'PM10']):
        df['PM25_PM10_Ratio'] = df['PM2.5'] / (df['PM10'] + 1e-8)
    
    if all(col in df.columns for col in ['NO2', 'O3']):
        df['NO2_O3_Ratio'] = df['NO2'] / (df['O3'] + 1e-8)
    
    return df

def create_lag_features(df, group_col, target_cols, lag_days=[1, 2, 7, 30]):
    print(f"🔧 Engineering lag features for {len(lag_days)} time steps...")
    df_out = df.copy()
    df_out = df_out.sort_values([group_col, 'Date'])
    for lag in lag_days:
        for col in target_cols:
            lag_col_name = f"{col}_lag_{lag}"
            df_out[lag_col_name] = df_out.groupby(group_col)[col].shift(lag)
    print("✅ Lag features created.")
    return df_out

def get_numerical_features(df, primary_features):
    """Get only numerical features from the dataframe"""
    numerical_features = []
    
    # Primary pollutants (should be numerical)
    numerical_features.extend([f for f in primary_features if f in df.columns])
    
    # Secondary pollutants
    secondary_features = ['NO', 'NOx', 'NH3', 'Benzene', 'Toluene', 'Xylene']
    numerical_features.extend([f for f in secondary_features if f in df.columns])
    
    # Meteorological features
    meteorological_features = ['Temperature', 'Humidity', 'Pressure', 'Wind_Speed']
    numerical_features.extend([f for f in meteorological_features if f in df.columns])
    
    # Temporal features (already encoded as numerical)
    temporal_features = [
        'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'IsWeekend',
        'IsWinter', 'IsSummer', 'IsMonsoon'
    ]
    numerical_features.extend([f for f in temporal_features if f in df.columns])
    
    # AQI enhanced features (created as numerical)
    numerical_features.extend([f for f in df.columns if 'AQI_' in f or '_Ratio' in f])
    
    # Lag features (should be numerical)
    numerical_features.extend([f for f in df.columns if '_lag_' in f])
    
    # Remove any potential duplicates and non-existent columns
    numerical_features = [f for f in set(numerical_features) if f in df.columns]
    
    # Verify they are numerical
    numerical_dtypes = df[numerical_features].dtypes
    non_numerical = numerical_dtypes[numerical_dtypes == 'object'].index.tolist()
    if non_numerical:
        print(f"⚠️  Removing non-numerical features: {non_numerical}")
        numerical_features = [f for f in numerical_features if f not in non_numerical]
    
    return sorted(numerical_features)

def create_sequences(df_scaled, all_features, primary_features, sequence_length=24, prediction_horizon=24, stride=1, max_sequences_per_city=1000):
    print(f"📊 Creating sequences from {len(df_scaled)} scaled records (Stride: {stride}, Max: {max_sequences_per_city})...")
    sequences = []
    targets = []
    target_indices = [all_features.index(col) for col in primary_features]
    
    for city, city_data in df_scaled.sort_values('City').groupby('City'):
        city_data = city_data.sort_values('Date')
        if len(city_data) < sequence_length + prediction_horizon:
            continue
        values = city_data[all_features].values
        city_sequences = []
        city_targets = []
        
        for i in range(0, len(values) - sequence_length - prediction_horizon, stride):
            sequence = values[i : i + sequence_length]
            target = values[i + sequence_length : i + sequence_length + prediction_horizon, target_indices]
            city_sequences.append(sequence)
            city_targets.append(target)
            if len(city_sequences) >= max_sequences_per_city:
                break
                
        if city_sequences:
            sequences.extend(city_sequences)
            targets.extend(city_targets)
            
    if len(sequences) == 0:
        print("❌ No sequences could be created - not enough data")
        return np.array([]), np.array([])
        
    sequences_np = np.array(sequences)
    targets_np = np.array(targets)
    print(f"📊 Created {len(sequences_np)} sequences")
    print(f"📊 Sequence shape: {sequences_np.shape}")
    print(f"📊 Target shape: {targets_np.shape}")
    return sequences_np, targets_np

def inverse_transform_predictions(predictions, scaling_info):
    original_shape = predictions.shape
    num_features = original_shape[-1]
    predictions_flat = predictions.reshape(-1, num_features)
    try:
        scaler = scaling_info.get('scaler')
        all_features = scaling_info.get('all_features')
        primary_features = scaling_info.get('primary_features')
        if not scaler or not all_features or not primary_features:
            raise ValueError("Scaling info is missing.")
        dummy_input = np.zeros((predictions_flat.shape[0], len(all_features)))
        primary_indices = [all_features.index(col) for col in primary_features]
        dummy_input[:, primary_indices] = predictions_flat
        predictions_inv = scaler.inverse_transform(dummy_input)
        predictions_orig = predictions_inv[:, primary_indices]
    except Exception as e:
        print(f"⚠️  Scaler inverse transform failed: {e}. Using fallback scaling.")
        # Simple fallback - assumes data was standardized
        predictions_orig = predictions_flat * 100 + 150
    return predictions_orig.reshape(original_shape)

def detailed_analysis(predictions, actuals, scaling_info):
    """Detailed analysis of prediction performance"""
    print("\n🔍 DETAILED PREDICTION ANALYSIS...")
    try:
        predictions_orig = inverse_transform_predictions(predictions, scaling_info)
        actuals_orig = inverse_transform_predictions(actuals, scaling_info)
        print("✅ Successfully inverse transformed predictions")
    except Exception as e:
        print(f"⚠️  Optimized inverse transform failed: {e}")
        print("🔄 Using direct scaled values for analysis")
        predictions_orig = predictions
        actuals_orig = actuals

    if 'AQI' in scaling_info['primary_features']:
        aqi_idx = scaling_info['primary_features'].index('AQI')
        aqi_predictions = predictions_orig[:, :, aqi_idx]
        aqi_actuals = actuals_orig[:, :, aqi_idx]
        
        print(f"📊 AQI PREDICTION STATISTICS:")
        print(f"   Predictions - Mean: {aqi_predictions.mean():.2f}, Std: {aqi_predictions.std():.2f}")
        print(f"   Actuals     - Mean: {aqi_actuals.mean():.2f}, Std: {aqi_actuals.std():.2f}")
        print(f"   Prediction range: [{aqi_predictions.min():.2f}, {aqi_predictions.max():.2f}]")
        print(f"   Actual range:     [{aqi_actuals.min():.2f}, {aqi_actuals.max():.2f}]")
        
        # Check for negative predictions
        negative_count = (aqi_predictions < 0).sum()
        if negative_count > 0:
            print(f"   ⚠️  {negative_count} negative AQI predictions detected!")
        
        # AQI category accuracy
        def aqi_category(aqi):
            if aqi <= 50: return 0
            elif aqi <= 100: return 1
            elif aqi <= 150: return 2
            elif aqi <= 200: return 3
            elif aqi <= 300: return 4
            else: return 5
        
        # Calculate category accuracy
        pred_cats = np.vectorize(aqi_category)(aqi_predictions.flatten())
        actual_cats = np.vectorize(aqi_category)(aqi_actuals.flatten())
        cat_accuracy = (pred_cats == actual_cats).mean()
        
        print(f"   AQI Category Accuracy: {cat_accuracy:.4f}")
        
        if aqi_predictions.std() < 10.0:
            print(f"⚠️  AQI predictions have low variance (Std: {aqi_predictions.std():.2f})")
            
    return predictions_orig, actuals_orig

def train_nf_vae_model():
    print("🚀 Starting PROVEN VAE Model Training...")
    data_loader = ComprehensiveAQIDataLoader()
    preprocessor = AQIPreprocessor()
    
    print("📥 Loading historical data...")
    df_raw = data_loader.load_historical_data()
    if df_raw is None or len(df_raw) == 0:
        print("❌ No data available for training")
        return None
        
    df_processed = preprocessor.preprocess_data(df_raw, for_training=True)
    if df_processed is None or len(df_processed) == 0:
        print("❌ No processed data available")
        return None
        
    print(f"📊 Processed data: {len(df_processed):,} records")

    # Enhanced feature engineering
    df_processed = enhance_aqi_features(df_processed)

    # Define primary features (what we want to predict)
    primary_features_base = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
    primary_features = [f for f in primary_features_base if f in df_processed.columns]

    if 'AQI' not in primary_features:
        print("❌ CRITICAL: 'AQI' not found in data. Cannot train model.")
        return None

    # Get only numerical features for modeling
    all_numerical_features = get_numerical_features(df_processed, primary_features)
    
    print(f"🔧 Using {len(all_numerical_features)} numerical features:")
    print(f"   Primary (output): {primary_features}")
    print(f"   Total numerical features: {len(all_numerical_features)}")

    print("🔪 Splitting data into training and validation sets by Time...")
    df_processed = df_processed.sort_values('Date')
    train_df, val_df = train_test_split(df_processed, test_size=0.2, shuffle=False)

    print(f"📊 Training samples (pre-sequence): {len(train_df)}")
    print(f"📊 Validation samples (pre-sequence): {len(val_df)}")
    if len(train_df) > 0 and len(val_df) > 0:
        print(f"   Train Date Range: {train_df['Date'].min()} to {train_df['Date'].max()}")
        print(f"   Valid Date Range: {val_df['Date'].min()} to {val_df['Date'].max()}")

    print("⚖️ Fitting StandardScaler on TRAINING data only...")
    scaler = StandardScaler()
    
    # Handle missing values only for numerical features
    imputation_values = train_df[all_numerical_features].median()
    train_df[all_numerical_features] = train_df[all_numerical_features].fillna(imputation_values)
    val_df[all_numerical_features] = val_df[all_numerical_features].fillna(imputation_values)
    
    # Scale only numerical features
    train_df[all_numerical_features] = scaler.fit_transform(train_df[all_numerical_features])
    val_df[all_numerical_features] = scaler.transform(val_df[all_numerical_features])
    print("✅ Scaler fitted and data transformed.")

    scaling_info = {
        'scaler': scaler,
        'all_features': all_numerical_features,
        'primary_features': primary_features,
        'imputation_values': imputation_values.to_dict()
    }

    train_sequences, train_targets = create_sequences(
        train_df, all_numerical_features, primary_features, stride=2, max_sequences_per_city=800
    )
    val_sequences, val_targets = create_sequences(
        val_df, all_numerical_features, primary_features, stride=2, max_sequences_per_city=800
    )

    if len(train_sequences) == 0:
        print("❌ No training sequences created.")
        return None

    train_data = torch.FloatTensor(train_sequences)
    train_targets = torch.FloatTensor(train_targets)
    val_data = torch.FloatTensor(val_sequences)
    val_targets = torch.FloatTensor(val_targets)

    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)

    # Use the proven model settings
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"📊 Final training sequences: {len(train_dataset)}")
    print(f"📊 Final validation sequences: {len(val_dataset)}")

    # Use PROVEN model hyperparameters
    input_dim = len(all_numerical_features)
    output_dim = len(primary_features)
    hidden_dim = 256
    latent_dim = 64
    num_layers = 2
    dropout_p = 0.2

    print(f"🎯 PROVEN Model Configuration:")
    print(f"   Input dimension: {input_dim}")
    print(f"   Output dimension: {output_dim}")
    print(f"   Hidden dimension: {hidden_dim}")
    print(f"   Latent dimension: {latent_dim}")
    print(f"   Num layers: {num_layers}")
    print(f"   Dropout: {dropout_p}")

    # Use the proven model
    model = SequentialVAE(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout_p=dropout_p
    )

    # Use the proven trainer - FIXED: Removed tf_anneal_epochs parameter
    trainer = GRU_VAE_Trainer(
        model,
        lr=1e-3,           # Higher learning rate for faster learning
        max_beta=0.01,     # Very low KL weight
        kl_anneal_epochs=20 # Quick KL warmup
    )

    os.makedirs('data/models', exist_ok=True)
    with open('data/models/nf_vae_scaling_info.pkl', 'wb') as f:
        pickle.dump(scaling_info, f)
    print("💾 Scaling info saved to data/models/nf_vae_scaling_info.pkl")

    # Use proven training settings
    epochs_to_run = 100
    patience_for_stopping = 5

    print("🤖 Training PROVEN VAE model...")
    try:
        final_val_loss, final_val_recon, final_val_kl = trainer.train(
            train_loader, val_loader, epochs=epochs_to_run, patience=patience_for_stopping
        )

        trainer.save_model('data/models/best_nf_vae.pth')

        print("📊 Evaluating VAE model...")
        trainer.model.eval()
        with torch.no_grad():
            val_predictions = []
            val_actuals = []
            for data, target in val_loader:
                data = data.to(trainer.device)
                target = target.to(trainer.device)
                target_seq_len = target.shape[1]
                # Use more samples for better evaluation
                predictions_samples = trainer.model.predict(data, target_seq_len, n_samples=10)
                predictions = torch.mean(predictions_samples, dim=1)
                val_predictions.append(predictions.cpu().numpy())
                val_actuals.append(target.cpu().numpy())

            val_predictions = np.concatenate(val_predictions)
            val_actuals = np.concatenate(val_actuals)

            try:
                val_predictions_orig, val_actuals_orig = detailed_analysis(
                    val_predictions, val_actuals, scaling_info
                )

                # Calculate metrics
                rmse_all = np.sqrt(np.mean((val_predictions_orig - val_actuals_orig) ** 2))
                mae_all = np.mean(np.abs(val_predictions_orig - val_actuals_orig))
                ss_res_all = np.sum((val_actuals_orig - val_predictions_orig) ** 2)
                ss_tot_all = np.sum((val_actuals_orig - np.mean(val_actuals_orig)) ** 2)
                r2_all = 1 - (ss_res_all / ss_tot_all) if ss_tot_all > 0 else 0

                if 'AQI' in scaling_info['primary_features']:
                    aqi_idx = scaling_info['primary_features'].index('AQI')
                    aqi_predictions = val_predictions_orig[:, :, aqi_idx]
                    aqi_actuals = val_actuals_orig[:, :, aqi_idx]
                    rmse_aqi = np.sqrt(np.mean((aqi_predictions - aqi_actuals) ** 2))
                    mae_aqi = np.mean(np.abs(aqi_predictions - aqi_actuals))
                    ss_res_aqi = np.sum((aqi_actuals - aqi_predictions) ** 2)
                    ss_tot_aqi = np.sum((aqi_actuals - np.mean(aqi_actuals)) ** 2)
                    r2_aqi = 1 - (ss_res_aqi / ss_tot_aqi) if ss_tot_aqi > 0 else 0
                else:
                    rmse_aqi = mae_aqi = r2_aqi = 0

            except Exception as e:
                print(f"⚠️  Evaluation metrics calculation failed: {e}")
                rmse_all = mae_all = r2_all = rmse_aqi = mae_aqi = r2_aqi = 0

        best_epoch = np.argmin(trainer.val_losses) if trainer.val_losses else 0
        best_val_loss = trainer.val_losses[best_epoch] if trainer.val_losses else final_val_loss

        performance = {
            'nf_vae': {
                'all_features': {
                    'RMSE': float(rmse_all),
                    'MAE': float(mae_all),
                    'R2': float(r2_all)
                },
                'aqi_only': {
                    'RMSE': float(rmse_aqi),
                    'MAE': float(mae_aqi),
                    'R2': float(r2_aqi)
                },
                'training_losses': {
                    'final_val_loss': float(final_val_loss),
                    'final_recon_loss': float(final_val_recon),
                    'final_kl_loss': float(final_val_kl),
                    'best_val_loss': float(best_val_loss),
                    'best_epoch': int(best_epoch + 1)
                }
            },
            'training_info': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'training_samples': len(train_dataset),
                'validation_samples': len(val_dataset),
                'input_dim': input_dim,
                'output_dim': output_dim,
                'sequence_length': 24,
                'prediction_horizon': 24,
                'features_used': scaling_info['all_features'],
                'primary_features': scaling_info['primary_features'],
                'model_type': 'SimpleVAE_Proven',
                'trainer_type': 'ProvenVAETrainer',
                'final_epoch': len(trainer.val_losses),
                'optimizations': {
                    'stride': 2,
                    'max_sequences_per_city': 800,
                    'batch_size': 32,
                    'hidden_dim': hidden_dim,
                    'latent_dim': latent_dim,
                    'num_layers': num_layers,
                    'dropout_p': dropout_p,
                    'epochs': epochs_to_run,
                    'patience': patience_for_stopping,
                    'max_beta': trainer.max_beta,
                    'kl_anneal_epochs': trainer.kl_anneal_epochs,
                    'enhanced_aqi_features': True
                }
            }
        }

        with open('data/models/nf_vae_performance.json', 'w') as f:
            json.dump(performance, f, indent=2)

        print(f"\n✅ VAE TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Final Validation Loss: {final_val_loss:.4f}")
        print(f"📊 Best Validation Loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
        print(f"\n📈 ALL FEATURES PERFORMANCE:")
        print(f"   RMSE: {rmse_all:.4f}")
        print(f"   MAE: {mae_all:.4f}")
        print(f"   R²: {r2_all:.4f}")

        if 'AQI' in scaling_info['primary_features']:
            print(f"\n🎯 AQI-SPECIFIC PERFORMANCE:")
            print(f"   RMSE: {rmse_aqi:.4f}")
            print(f"   MAE: {mae_aqi:.4f}")
            print(f"   R²: {r2_aqi:.4f}")
            if len(aqi_predictions) > 0:
                print(f"\n🔍 SAMPLE AQI PREDICTIONS VS ACTUALS:")
                print(f"   Predictions: {aqi_predictions[0, :5].round(2)}")
                print(f"   Actuals:     {aqi_actuals[0, :5].round(2)}")

        print(f"\n💾 MODEL ARTIFACTS SAVED:")
        print(f"   Model: data/models/best_nf_vae.pth")
        print(f"   Scaling info: data/models/nf_vae_scaling_info.pkl")
        print(f"   Performance: data/models/nf_vae_performance.json")

        return performance

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    train_nf_vae_model()