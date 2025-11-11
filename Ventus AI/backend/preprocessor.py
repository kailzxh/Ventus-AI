# backend/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class AQIPreprocessor:
    def __init__(self):
        """
        This preprocessor is now streamlined for feature engineering ONLY.
        Imputation and Scaling are handled in the training scripts to prevent data leakage.
        """
        pass
    
    def preprocess_data(self, df, for_training=True):
        """Main preprocessing pipeline with error handling"""
        print("🔧 Starting Streamlined Preprocessing...")
        
        if df is None or len(df) == 0:
            print("❌ No data to preprocess")
            return df
        
        try:
            # Make a copy to avoid modifying original data
            df_processed = df.copy()
            
            # Step 1: Handle critical missing values (Date, Categories)
            df_processed = self._handle_critical_missing(df_processed)
            
            # Step 2: Create temporal features
            df_processed = self._create_temporal_features(df_processed)
            
            # Step 3: Feature engineering
            df_processed = self._feature_engineering(df_processed)
            
            # Step 4: Handle outliers if for training
            if for_training:
                # Capping extreme outliers before scaling is a valid step
                df_processed = self._handle_outliers(df_processed)
            
            # Step 5: Final cleanup (no imputation)
            df_processed = self._final_cleanup(df_processed)
            
            print(f"✅ Preprocessing complete: {len(df_processed):,} records")
            return df_processed
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return df # Return original df as last resort
    
    def _handle_critical_missing(self, df):
        """
        Handle missing values for non-numerical or critical columns.
        *** Numerical NaNs are intentionally NOT filled to prevent data leakage. ***
        """
        print("   🔧 Handling critical missing values (Date, Categories)...")
        
        df = df.copy()
        
        # Handle Date column first
        if 'Date' in df.columns:
            # Convert to datetime and count invalid dates
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                print(f"   ⚠️  Found {invalid_dates:,} invalid dates (will be dropped by training script)")
        
        # Handle categorical data
        categorical_cols = ['City', 'Station', 'AQI_Bucket', 'AQI_Category']
        
        for col in categorical_cols:
            if col in df.columns:
                missing_before = df[col].isna().sum()
                if missing_before > 0:
                    df[col] = df[col].fillna('Unknown')
                    print(f"   🏷️  {col}: Filled {missing_before:,} missing values with 'Unknown'")
        
        return df

    def _create_temporal_features(self, df):
        """
        Create temporal features from date.
        *** This now creates ONLY the features used by train_models.py and train_nf_vae.py ***
        """
        print("   ⏰ Creating temporal features...")
        
        if 'Date' not in df.columns:
            print("   ⚠️  No Date column found for temporal features")
            return df
        
        df = df.copy()
        
        # Create temporal features only for valid dates
        valid_dates_mask = df['Date'].notna()
        if not valid_dates_mask.any():
            print("   ⚠️  No valid dates found. Skipping temporal features.")
            return df
            
        # --- Features for Baseline Models (train_models.py) ---
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(float)
        
        # --- Features for VAE Model (train_nf_vae.py) ---
        # Cyclical Month
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Cyclical Day of Week
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Seasonal flags
        df['IsWinter'] = df['Month'].isin([11, 12, 1, 2]).astype(float)
        df['IsSummer'] = df['Month'].isin([3, 4, 5, 6]).astype(float)
        df['IsMonsoon'] = df['Month'].isin([7, 8, 9, 10]).astype(float)
        
        # --- Fill NaNs for rows that had NaT dates ---
        # We fill with 0, a neutral value that won't break the scaler
        temporal_cols = [
            'Year', 'Month', 'DayOfWeek', 'IsWeekend', 
            'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
            'IsWinter', 'IsSummer', 'IsMonsoon'
        ]
        for col in temporal_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0) 
        
        print("   ✅ Temporal features created.")
        return df

    def _feature_engineering(self, df):
        """Create additional features for better predictions"""
        print("   🔧 Creating additional features...")
        
        df = df.copy()
        
        # Pollutant ratios and combinations
        if all(col in df.columns for col in ['PM2.5', 'PM10']):
            # Ensure no division by zero
            df['PM_Ratio'] = df['PM2.5'] / (df['PM10'] + 1e-6) # Add epsilon
            df['PM_Ratio'] = df['PM_Ratio'].replace([np.inf, -np.inf], np.nan)
            # Fill NaNs with a reasonable default (0.5)
            df['PM_Ratio'] = df['PM_Ratio'].fillna(0.5)
        
        # Air quality indices
        if 'AQI' in df.columns:
            def get_aqi_category(aqi):
                if pd.isna(aqi): return 'Unknown'
                aqi = float(aqi)
                if aqi <= 50: return 'Good'
                elif aqi <= 100: return 'Satisfactory'
                elif aqi <= 200: return 'Moderate'
                elif aqi <= 300: return 'Poor'
                elif aqi <= 400: return 'Very Poor'
                else: return 'Severe'
            
            df['AQI_Category'] = df['AQI'].apply(get_aqi_category)
            
            def get_aqi_severity(aqi):
                if pd.isna(aqi): return 2  # Default to Moderate
                aqi = float(aqi)
                if aqi <= 50: return 0
                elif aqi <= 100: return 1
                elif aqi <= 200: return 2
                elif aqi <= 300: return 3
                elif aqi <= 400: return 4
                else: return 5
            
            df['AQI_Severity'] = df['AQI'].apply(get_aqi_severity)
        
        # Meteorological interactions
        if all(col in df.columns for col in ['Temperature', 'Humidity']):
            df['Temp_Humidity_Interaction'] = df['Temperature'] * df['Humidity'] / 100
        
        # *** REMOVED City_Code. This is handled by train_models.py (LabelEncoder) ***
        
        print("   ✅ Additional features created")
        return df
        
    def _handle_outliers(self, df):
        """Handle outliers in numerical columns"""
        print("   📊 Handling outliers...")
        
        numerical_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI', 
                          'Temperature', 'Humidity', 'Pressure', 'Wind_Speed']
        
        outlier_count = 0
        for col in numerical_cols:
            if col in df.columns:
                # Use IQR method for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Only cap if IQR is valid
                if pd.isna(IQR) or IQR == 0:
                    continue
                    
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count += outliers_mask.sum()
                
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        if outlier_count > 0:
            print(f"   🔧 Capped {outlier_count:,} outliers")
        
        return df
    
    def _final_cleanup(self, df):
        """Final data cleanup and validation"""
        print("   🧹 Final cleanup...")
        
        # Remove any remaining infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # *** REMOVED: Numerical NaN filling. This is data leakage. ***
        
        # Ensure categorical columns have no NaN (already done, but safe)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna('Unknown')
        
        # Remove completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            print(f"   🗑️  Removed {len(empty_cols)} empty columns: {empty_cols}")
        
        print(f"   ✅ Final cleanup complete: {len(df)} records, {len(df.columns)} columns")
        return df

# ***
# *** ALL DEAD CODE BELOW THIS LINE HAS BEEN DELETED ***
# - _encode_categorical_features (unused)
# - _create_lag_features (unused, slow)
# - _create_rolling_features (unused, slow)
# - _scale_features (unused, data leakage)
# - get_feature_columns (unused)
# - prepare_prediction_data (unused)
# - The duplicate _handle_outliers function
# ***