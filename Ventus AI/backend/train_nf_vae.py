# backend/train_nf_vae.py
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import sys
import pickle

# Add the current directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.nf_vae import EnhancedNFVAE, ComprehensiveNFVAETrainer
from data_loader import ComprehensiveAQIDataLoader
from preprocessor import AQIPreprocessor

def prepare_comprehensive_nf_vae_data_optimized(historical_data, sequence_length=24, prediction_horizon=24, stride=2, max_sequences_per_city=500):
    """Optimized version with reduced sequence count and faster processing"""
    print("📊 Preparing OPTIMIZED NF-VAE training data...")
    
    # Define feature hierarchy - use ALL available features
    primary_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
    secondary_features = ['NO', 'NOx', 'NH3', 'Benzene', 'Toluene', 'Xylene']
    meteorological_features = ['Temperature', 'Humidity', 'Pressure', 'Wind_Speed', 'Wind_Direction']
    temporal_features = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 
                        'day_of_week_sin', 'day_of_week_cos', 'is_weekend',
                        'season_spring', 'season_summer', 'season_fall', 'season_winter']
    
    # Collect all available features
    all_possible_features = primary_features + secondary_features + meteorological_features + temporal_features
    available_features = [col for col in all_possible_features if col in historical_data.columns]
    
    print(f"🔧 Using {len(available_features)} available features:")
    print(f"   Primary: {[f for f in primary_features if f in available_features]}")
    print(f"   Secondary: {[f for f in secondary_features if f in available_features]}")
    print(f"   Meteorological: {[f for f in meteorological_features if f in available_features]}")
    print(f"   Temporal: {[f for f in temporal_features if f in available_features]}")
    
    # Group by city and create sequences
    sequences = []
    targets = []
    
    # Use single scaler for faster processing
    scaler = StandardScaler()
    
    # Store scaling info
    scaling_info = {
        'scaler': scaler,
        'all_features': available_features,
        'primary_features': [f for f in primary_features if f in available_features],
        'secondary_features': [f for f in secondary_features if f in available_features],
        'meteo_features': [f for f in meteorological_features if f in available_features],
        'temporal_features': [f for f in temporal_features if f in available_features],
        'pollutant_features': [f for f in primary_features + secondary_features if f in available_features and f != 'AQI']
    }
    
    # First pass: fit scaler on sample data for speed
    print("   🔄 Fitting scaler on sample data...")
    sample_size = min(20000, len(historical_data))
    sample_data = historical_data[available_features].sample(sample_size, random_state=42)
    sample_data = sample_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler.fit(sample_data)
    
    # Second pass: transform data and create sequences with optimizations
    print("   🔄 Creating optimized sequences...")
    cities_processed = 0
    
    for city in historical_data['City'].unique():
        city_data = historical_data[historical_data['City'] == city].copy()
        city_data = city_data.sort_values('Date')
        
        # Skip cities with insufficient data
        if len(city_data) < sequence_length + prediction_horizon + 10:
            continue
            
        # Select features and ensure they're numeric
        city_features = city_data[available_features].apply(pd.to_numeric, errors='coerce')
        city_features = city_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Apply scaling
        values = scaler.transform(city_features)
        
        # Create sequences with stride and limit
        city_sequences = []
        city_targets = []
        
        for i in range(0, len(values) - sequence_length - prediction_horizon, stride):
            sequence = values[i:i + sequence_length]
            # Target includes only primary pollutants + AQI
            target_indices = [available_features.index(col) for col in primary_features if col in available_features]
            target = values[i + sequence_length:i + sequence_length + prediction_horizon, target_indices]
            
            city_sequences.append(sequence)
            city_targets.append(target)
            
            # Limit sequences per city to prevent domination by large cities
            if len(city_sequences) >= max_sequences_per_city:
                break
        
        if city_sequences:
            sequences.extend(city_sequences)
            targets.extend(city_targets)
            cities_processed += 1
            print(f"      {city}: {len(city_sequences)} sequences")
        
        # Early stop if we have enough cities
        if cities_processed >= 30:  # Limit to top 30 cities
            print(f"   ⏹️  Limited to top {cities_processed} cities for efficiency")
            break
    
    if len(sequences) == 0:
        print("❌ No sequences could be created - not enough data")
        return np.array([]), np.array([]), None
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    print(f"📊 Created {len(sequences)} sequences (optimized)")
    print(f"📊 Sequence shape: {sequences.shape}")  # (n_sequences, 24, total_features)
    print(f"📊 Target shape: {targets.shape}")      # (n_sequences, 24, primary_features)
    print(f"📊 Input features: {len(available_features)}")
    print(f"📊 Output features: {len([f for f in primary_features if f in available_features])}")
    print(f"📊 Sequence range: [{sequences.min():.3f}, {sequences.max():.3f}]")
    print(f"📊 Target range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"📊 Cities processed: {cities_processed}")
    
    return sequences, targets, scaling_info

def inverse_transform_predictions_optimized(predictions, scaling_info):
    """Optimized inverse transform using scaler if available"""
    predictions_flat = predictions.reshape(-1, predictions.shape[-1])
    
    try:
        # Try to use the scaler for inverse transform
        if 'scaler' in scaling_info:
            # Create a dummy array with all features and then extract primary features
            dummy_input = np.zeros((predictions_flat.shape[0], len(scaling_info['all_features'])))
            primary_indices = [scaling_info['all_features'].index(col) for col in scaling_info['primary_features']]
            
            for i, idx in enumerate(primary_indices):
                dummy_input[:, idx] = predictions_flat[:, i]
            
            # Inverse transform
            predictions_inv = scaling_info['scaler'].inverse_transform(dummy_input)
            predictions_orig = predictions_inv[:, primary_indices]
        else:
            # Fallback to manual calculation
            predictions_orig = inverse_transform_predictions_simple(predictions, scaling_info)
            
    except Exception as e:
        print(f"⚠️  Scaler inverse transform failed, using manual: {e}")
        predictions_orig = inverse_transform_predictions_simple(predictions, scaling_info)
    
    return predictions_orig.reshape(predictions.shape)

def inverse_transform_predictions_simple(predictions, scaling_info):
    """Simplified inverse transform using manual calculations"""
    predictions_flat = predictions.reshape(-1, predictions.shape[-1])
    predictions_orig = np.zeros_like(predictions_flat)
    
    # Define reasonable ranges for each pollutant (based on typical values)
    pollutant_ranges = {
        'PM2.5': (0, 500),
        'PM10': (0, 600), 
        'NO2': (0, 200),
        'SO2': (0, 100),
        'CO': (0, 10),
        'O3': (0, 200),
        'AQI': (0, 500)
    }
    
    primary_features = scaling_info['primary_features']
    
    for i, feature in enumerate(primary_features):
        feature_data = predictions_flat[:, i]
        
        if feature in pollutant_ranges:
            min_val, max_val = pollutant_ranges[feature]
            # Assuming data was scaled to roughly [-2, 2] range by StandardScaler
            # Adjust based on your actual data distribution
            predictions_orig[:, i] = np.clip(feature_data * 100 + 150, min_val, max_val)
        else:
            # Default scaling for unknown features
            predictions_orig[:, i] = feature_data * 100 + 150
    
    return predictions_orig.reshape(predictions.shape)

def analyze_predictions(val_predictions, val_actuals, scaling_info):
    """Analyze predictions comprehensively"""
    print("\n🔍 Analyzing Prediction Performance...")
    
    try:
        # Try the optimized inverse transform first
        val_predictions_orig = inverse_transform_predictions_optimized(val_predictions, scaling_info)
        val_actuals_orig = inverse_transform_predictions_optimized(val_actuals, scaling_info)
        
        print("✅ Successfully inverse transformed predictions")
        
    except Exception as e:
        print(f"⚠️  Optimized inverse transform failed: {e}")
        print("🔄 Using direct scaled values for analysis")
        val_predictions_orig = val_predictions
        val_actuals_orig = val_actuals
    
    # AQI-specific analysis
    if 'AQI' in scaling_info['primary_features']:
        aqi_idx = scaling_info['primary_features'].index('AQI')
        aqi_predictions = val_predictions_orig[:, :, aqi_idx]
        aqi_actuals = val_actuals_orig[:, :, aqi_idx]
        
        print(f"📊 AQI Prediction Statistics:")
        print(f"   Predictions - Mean: {aqi_predictions.mean():.2f}, Std: {aqi_predictions.std():.2f}")
        print(f"   Actuals     - Mean: {aqi_actuals.mean():.2f}, Std: {aqi_actuals.std():.2f}")
        print(f"   Prediction range: [{aqi_predictions.min():.2f}, {aqi_predictions.max():.2f}]")
        print(f"   Actual range:     [{aqi_actuals.min():.2f}, {aqi_actuals.max():.2f}]")
        
        # Check if predictions are constant
        if aqi_predictions.std() < 1.0:
            print("⚠️  AQI predictions are nearly constant")
    
    return val_predictions_orig, val_actuals_orig

def train_nf_vae_model():
    """Train the NF-VAE model with comprehensive features - OPTIMIZED VERSION"""
    print("🚀 Starting OPTIMIZED NF-VAE Model Training...")
    
    # Load data using comprehensive loader
    data_loader = ComprehensiveAQIDataLoader()
    preprocessor = AQIPreprocessor()
    
    print("📥 Loading historical data...")
    df_raw = data_loader.load_historical_data()
    
    if df_raw is None or len(df_raw) == 0:
        print("❌ No data available for training")
        return None
    
    # Preprocess data with enhanced features
    df_processed = preprocessor.preprocess_data(df_raw, for_training=True)
    
    if df_processed is None or len(df_processed) == 0:
        print("❌ No processed data available")
        return None
    
    print(f"📊 Processed data: {len(df_processed):,} records")
    
    # Prepare OPTIMIZED sequences for NF-VAE
    sequences, targets, scaling_info = prepare_comprehensive_nf_vae_data_optimized(
        df_processed, 
        stride=2,                    # Skip every other sequence
        max_sequences_per_city=400   # Limit sequences per city
    )
    
    if len(sequences) == 0:
        print("❌ No sequences created for training")
        return None
    
    # Verify no NaN in sequences
    if np.isnan(sequences).any() or np.isnan(targets).any():
        print("⚠️  NaN values detected in sequences - cleaning...")
        sequences = np.nan_to_num(sequences, nan=0.0)
        targets = np.nan_to_num(targets, nan=0.0)
    
    # Check for infinite values
    if np.isinf(sequences).any() or np.isinf(targets).any():
        print("⚠️  Infinite values detected in sequences - cleaning...")
        sequences = np.nan_to_num(sequences, posinf=3.0, neginf=-3.0)
        targets = np.nan_to_num(targets, posinf=3.0, neginf=-3.0)
    
    print(f"✅ Data validation passed - No NaN or Inf values")
    
    # Split data
    split_idx = int(0.8 * len(sequences))
    train_sequences = sequences[:split_idx]
    train_targets = targets[:split_idx]
    val_sequences = sequences[split_idx:]
    val_targets = targets[split_idx:]
    
    print(f"📊 Training sequences: {len(train_sequences)}")
    print(f"📊 Validation sequences: {len(val_sequences)}")
    
    # Convert to tensors
    train_data = torch.FloatTensor(train_sequences)
    train_targets = torch.FloatTensor(train_targets)
    val_data = torch.FloatTensor(val_sequences)
    val_targets = torch.FloatTensor(val_targets)
    
    # Verify tensor values
    if torch.isnan(train_data).any() or torch.isnan(train_targets).any():
        print("⚠️  NaN in training tensors - replacing with safe values")
        train_data = torch.nan_to_num(train_data, nan=0.0)
        train_targets = torch.nan_to_num(train_targets, nan=0.0)
    
    if torch.isnan(val_data).any() or torch.isnan(val_targets).any():
        print("⚠️  NaN in validation tensors - replacing with safe values")
        val_data = torch.nan_to_num(val_data, nan=0.0)
        val_targets = torch.nan_to_num(val_targets, nan=0.0)
    
    # Create data loaders with LARGER batch sizes
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)   # Increased from 32
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)      # Increased from 32
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")
    
    # Initialize OPTIMIZED model with appropriate dimensions
    input_dim = sequences.shape[2]  # Number of input features
    output_dim = targets.shape[2]   # Number of output features (primary pollutants)
    
    print(f"🎯 Model input dimension: {input_dim}")
    print(f"🎯 Model output dimension: {output_dim}")
    
    # Use smaller model for faster training
    model = EnhancedNFVAE(
        input_dim=input_dim,
        hidden_dim=128,  # Reduced from 256 for speed
        latent_dim=16,   # Reduced from 32 for speed
        sequence_length=24,
        num_pollutants=output_dim
    )
    
    # Initialize comprehensive trainer
    trainer = ComprehensiveNFVAETrainer(model, aqi_weight=2.0)
    
    # Create models directory
    os.makedirs('data/models', exist_ok=True)
    
    # Save scaling info for future use
    with open('data/models/nf_vae_scaling_info.pkl', 'wb') as f:
        pickle.dump(scaling_info, f)
    print("💾 Scaling info saved to data/models/nf_vae_scaling_info.pkl")
    
    # Train model with FEWER epochs
    print("🤖 Training OPTIMIZED NF-VAE model...")
    try:
        # Train with fewer epochs and more patience
        final_val_loss, final_val_recon, final_val_kl = trainer.train(
            train_loader, val_loader, epochs=50, patience=15  # Reduced epochs, increased patience
        )
        
        # Save final model
        trainer.save_model('data/models/best_nf_vae.pth')
        
        # Evaluate model
        print("📊 Evaluating NF-VAE model...")
        
        # Calculate additional metrics
        trainer.model.eval()
        with torch.no_grad():
            val_predictions = []
            val_actuals = []
            
            for data, target in val_loader:
                # Move data to the same device as model
                data = data.to(trainer.device)
                target = target.to(trainer.device)
                
                predictions = trainer.model.predict(data)
                val_predictions.append(predictions.cpu().numpy())
                val_actuals.append(target.cpu().numpy())
            
            val_predictions = np.concatenate(val_predictions)
            val_actuals = np.concatenate(val_actuals)
            
            try:
                # Analyze predictions and inverse transform
                val_predictions_orig, val_actuals_orig = analyze_predictions(
                    val_predictions, val_actuals, scaling_info
                )
                
                # Calculate RMSE and MAE for all features
                rmse_all = np.sqrt(np.mean((val_predictions_orig - val_actuals_orig) ** 2))
                mae_all = np.mean(np.abs(val_predictions_orig - val_actuals_orig))
                
                # Calculate R² for all features
                ss_res_all = np.sum((val_actuals_orig - val_predictions_orig) ** 2)
                ss_tot_all = np.sum((val_actuals_orig - np.mean(val_actuals_orig)) ** 2)
                r2_all = 1 - (ss_res_all / ss_tot_all) if ss_tot_all > 0 else 0
                
                # Calculate metrics specifically for AQI
                if 'AQI' in scaling_info['primary_features']:
                    aqi_idx = scaling_info['primary_features'].index('AQI')
                    aqi_predictions = val_predictions_orig[:, :, aqi_idx]
                    aqi_actuals = val_actuals_orig[:, :, aqi_idx]
                    
                    rmse_aqi = np.sqrt(np.mean((aqi_predictions - aqi_actuals) ** 2))
                    mae_aqi = np.mean(np.abs(aqi_predictions - aqi_actuals))
                    
                    # Calculate R² for AQI
                    ss_res_aqi = np.sum((aqi_actuals - aqi_predictions) ** 2)
                    ss_tot_aqi = np.sum((aqi_actuals - np.mean(aqi_actuals)) ** 2)
                    r2_aqi = 1 - (ss_res_aqi / ss_tot_aqi) if ss_tot_aqi > 0 else 0
                else:
                    rmse_aqi = mae_aqi = r2_aqi = 0
                    
            except Exception as e:
                print(f"⚠️  Evaluation metrics calculation failed: {e}")
                print("🔄 Using scaled values for metrics")
                # Use scaled values if inverse transform fails
                rmse_all = np.sqrt(np.mean((val_predictions - val_actuals) ** 2))
                mae_all = np.mean(np.abs(val_predictions - val_actuals))
                r2_all = 0
                rmse_aqi = mae_aqi = r2_aqi = 0
                
                # Still try to get AQI stats if possible
                if 'AQI' in scaling_info['primary_features']:
                    aqi_idx = scaling_info['primary_features'].index('AQI')
                    aqi_predictions = val_predictions[:, :, aqi_idx]
                    aqi_actuals = val_actuals[:, :, aqi_idx]
                    rmse_aqi = np.sqrt(np.mean((aqi_predictions - aqi_actuals) ** 2))
                    mae_aqi = np.mean(np.abs(aqi_predictions - aqi_actuals))
        
        # Find best epoch from training history
        best_epoch = np.argmin(trainer.val_losses) if trainer.val_losses else 0
        best_val_loss = trainer.val_losses[best_epoch] if trainer.val_losses else final_val_loss
        
        # Save performance metrics
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
                'model_type': 'EnhancedNFVAE_Optimized',
                'trainer_type': 'ComprehensiveNFVAETrainer',
                'final_epoch': len(trainer.val_losses),
                'aqi_weight': 2.0,
                'optimizations': {
                    'stride': 2,
                    'max_sequences_per_city': 400,
                    'batch_size': 64,
                    'hidden_dim': 128,
                    'latent_dim': 16,
                    'epochs': 50,
                    'patience': 15
                }
            }
        }
        
        with open('data/models/nf_vae_performance.json', 'w') as f:
            json.dump(performance, f, indent=2)
        
        print(f"✅ OPTIMIZED NF-VAE training completed!")
        print(f"📊 Final Validation Loss: {final_val_loss:.4f}")
        print(f"📊 Final Recon Loss: {final_val_recon:.4f}")
        print(f"📊 Final KL Loss: {final_val_kl:.4f}")
        print(f"📊 Best Validation Loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
        print(f"\n📈 All Features Performance:")
        print(f"   RMSE: {rmse_all:.4f}")
        print(f"   MAE: {mae_all:.4f}")
        print(f"   R²: {r2_all:.4f}")
        
        if 'AQI' in scaling_info['primary_features']:
            print(f"\n🎯 AQI-Specific Performance:")
            print(f"   RMSE: {rmse_aqi:.4f}")
            print(f"   MAE: {mae_aqi:.4f}")
            print(f"   R²: {r2_aqi:.4f}")
            
            # Print sample predictions vs actuals
            if len(aqi_predictions) > 0:
                print(f"\n🔍 Sample AQI Predictions vs Actuals:")
                print(f"   First 5 predictions: {aqi_predictions[0, :5].round(2)}")
                print(f"   First 5 actuals:     {aqi_actuals[0, :5].round(2)}")
        
        # Print training summary
        print(f"\n📊 Training Summary:")
        print(f"   Best validation loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
        print(f"   Final validation loss: {final_val_loss:.4f}")
        print(f"   Total epochs trained: {len(trainer.val_losses)}")
        print(f"   Input features: {input_dim}")
        print(f"   Output features: {output_dim}")
        print(f"   Model saved to: data/models/best_nf_vae.pth")
        print(f"   Scaling info saved to: data/models/nf_vae_scaling_info.pkl")
        print(f"   Performance saved to: data/models/nf_vae_performance.json")
        
        # Print optimization summary
        print(f"\n⚡ Optimization Summary:")
        print(f"   Batch size: 64 (was 32)")
        print(f"   Sequence stride: 2 (skip every other sequence)")
        print(f"   Max sequences per city: 400")
        print(f"   Model hidden dim: 128 (was 256)")
        print(f"   Model latent dim: 16 (was 32)")
        print(f"   Epochs: 50 (was 100)")
        print(f"   Patience: 15 (was 20)")
        
        return performance
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # Train the model
    train_nf_vae_model()