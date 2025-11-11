# backend/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class AQIPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputer = None
        
    def preprocess_data(self, df, for_training=True):
        """Main preprocessing pipeline with error handling"""
        print("🔧 Starting Advanced Preprocessing...")
        
        if df is None or len(df) == 0:
            print("❌ No data to preprocess")
            return df
        
        try:
            # Make a copy to avoid modifying original data
            df_processed = df.copy()
            
            # Step 1: Handle missing values
            df_processed = self._handle_missing_values(df_processed)
            
            # Step 2: Create temporal features (with proper NA handling)
            df_processed = self._create_temporal_features(df_processed)
            
            # Step 3: Feature engineering
            df_processed = self._feature_engineering(df_processed)
            
            # Step 4: Handle outliers if for training
            if for_training:
                df_processed = self._handle_outliers(df_processed)
            
            # Step 5: Final cleanup
            df_processed = self._final_cleanup(df_processed)
            
            print(f"✅ Preprocessing complete: {len(df_processed):,} records")
            return df_processed
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: basic preprocessing that won't fail
            print("🔄 Attempting basic preprocessing fallback...")
            try:
                df_fallback = df.copy()
                # Just handle missing values and return
                df_fallback = self._handle_missing_values(df_fallback)
                print(f"✅ Basic preprocessing fallback complete: {len(df_fallback):,} records")
                return df_fallback
            except Exception as fallback_error:
                print(f"❌ Fallback preprocessing also failed: {fallback_error}")
                # Return original data as last resort
                return df
    
    def _handle_missing_values(self, df):
        """Handle missing values with improved strategy"""
        print("   🔧 Handling missing values...")
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Handle Date column first
        if 'Date' in df.columns:
            # Convert to datetime and count invalid dates
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                print(f"   ⚠️  Found {invalid_dates:,} invalid dates (converted to NaT)")
        
        # Define columns by type for different imputation strategies
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
        meteorological_cols = ['Temperature', 'Humidity', 'Pressure', 'Wind_Speed']
        categorical_cols = ['City', 'Station', 'AQI_Bucket', 'AQI_Category']
        
        # Initialize missing counts
        total_missing_before = df.isnull().sum().sum()
        
        # Handle pollutants - forward fill then backward fill, then median
        for col in pollutant_cols:
            if col in df.columns:
                missing_before = df[col].isna().sum()
                if missing_before > 0:
                    # Try forward fill (carry last observation forward)
                    df[col] = df[col].fillna(method='ffill')
                    # Try backward fill (carry next observation backward)
                    df[col] = df[col].fillna(method='bfill')
                    # Remaining missing values with median
                    remaining_missing = df[col].isna().sum()
                    if remaining_missing > 0:
                        df[col] = df[col].fillna(df[col].median())
                    missing_after = df[col].isna().sum()
                    print(f"      🧹 {col}: {missing_before:,} → {missing_after:,} missing")
        
        # Handle meteorological data - similar strategy
        for col in meteorological_cols:
            if col in df.columns:
                missing_before = df[col].isna().sum()
                if missing_before > 0:
                    df[col] = df[col].fillna(method='ffill')
                    df[col] = df[col].fillna(method='bfill')
                    remaining_missing = df[col].isna().sum()
                    if remaining_missing > 0:
                        df[col] = df[col].fillna(df[col].median())
                    missing_after = df[col].isna().sum()
                    print(f"      🌡️  {col}: {missing_before:,} → {missing_after:,} missing")
        
        # Handle categorical data
        for col in categorical_cols:
            if col in df.columns:
                missing_before = df[col].isna().sum()
                if missing_before > 0:
                    df[col] = df[col].fillna('Unknown')
                    missing_after = df[col].isna().sum()
                    print(f"      🏷️  {col}: {missing_before:,} → {missing_after:,} missing")
        
        total_missing_after = df.isnull().sum().sum()
        print(f"   ✅ Missing values handled: {total_missing_before:,} → {total_missing_after:,}")
        
        return df
    def _create_temporal_features(self, df):
        """Create temporal features from date with proper NA handling"""
        print("   ⏰ Creating temporal features...")
        
        if 'Date' not in df.columns:
            print("   ⚠️  No Date column found for temporal features")
            return df
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Ensure Date is datetime and handle NaT values
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Create temporal features only for valid dates
        valid_dates_mask = df['Date'].notna()
        
        if valid_dates_mask.any():
            # For valid dates, create temporal features
            valid_dates = df.loc[valid_dates_mask, 'Date']
            
            # Year, Month, Day - these work with NaT
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['DayOfYear'] = df['Date'].dt.dayofyear
            df['Quarter'] = df['Date'].dt.quarter
            
            # WeekOfYear - handle carefully to avoid NA conversion issues
            df['WeekOfYear'] = np.nan  # Initialize with NaN
            try:
                # Use isocalendar but handle potential NA values
                week_values = valid_dates.dt.isocalendar().week
                df.loc[valid_dates_mask, 'WeekOfYear'] = week_values
            except Exception as e:
                print(f"   ⚠️  Week calculation failed: {e}")
                # Fallback: calculate week manually
                df.loc[valid_dates_mask, 'WeekOfYear'] = (df.loc[valid_dates_mask, 'DayOfYear'] - 1) // 7 + 1
            
            # Seasonal features
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(float)
            df['IsWinter'] = df['Month'].isin([11, 12, 1, 2]).astype(float)
            df['IsSummer'] = df['Month'].isin([3, 4, 5, 6]).astype(float)
            df['IsMonsoon'] = df['Month'].isin([7, 8, 9, 10]).astype(float)
            
            # Cyclical features for month and day of week
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
            
            print(f"   ✅ Created temporal features for {valid_dates_mask.sum():,} valid dates")
            
            # Report invalid dates
            invalid_dates_count = (~valid_dates_mask).sum()
            if invalid_dates_count > 0:
                print(f"   ⚠️  Found {invalid_dates_count:,} records with invalid dates (temporal features will be NaN)")
        else:
            print("   ⚠️  No valid dates found for temporal features")
        
        return df
    
    def _encode_categorical_features(self, df):
        """Encode categorical variables"""
        print("   🏷️ Encoding categorical features...")
        
        if len(df) == 0:
            return df
            
        if 'City' in df.columns:
            self.encoders['city'] = LabelEncoder()
            df['City_Encoded'] = self.encoders['city'].fit_transform(df['City'])
        
        if 'Season' in df.columns:
            self.encoders['season'] = LabelEncoder()
            df['Season_Encoded'] = self.encoders['season'].fit_transform(df['Season'])
        
        if 'AQI_Bucket' in df.columns:
            self.encoders['aqi_bucket'] = LabelEncoder()
            df['AQI_Bucket_Encoded'] = self.encoders['aqi_bucket'].fit_transform(df['AQI_Bucket'])
        
        return df
    
    def _create_lag_features(self, df):
        """Create lagged features for time series"""
        print("   📈 Creating lag features...")
        
        if len(df) == 0:
            return df
            
        # Check if we have the necessary columns
        if 'City' not in df.columns or 'Date' not in df.columns:
            print("   ⚠️  Missing City or Date columns for lag features")
            return df
            
        df = df.sort_values(['City', 'Date'])
        
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        if not available_pollutants:
            print("   ⚠️  No pollutant columns found for lag features")
            return df
            
        lag_periods = [1, 2, 3, 7]  # Reduced from [1, 2, 3, 7, 14, 30] to avoid too many NaNs
        
        for city in df['City'].unique():
            city_mask = df['City'] == city
            city_data = df[city_mask]
            
            # Only create lags if we have enough data for this city
            if len(city_data) > max(lag_periods):
                for col in available_pollutants:
                    for lag in lag_periods:
                        df.loc[city_mask, f'{col}_lag_{lag}'] = df.loc[city_mask, col].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df):
        """Create rolling statistics"""
        print("   📊 Creating rolling features...")
        
        if len(df) == 0:
            return df
            
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        if not available_pollutants:
            print("   ⚠️  No pollutant columns found for rolling features")
            return df
            
        for col in available_pollutants:
            # Use min_periods=1 to avoid creating too many NaNs
            df[f'{col}_rolling_mean_7'] = df.groupby('City')[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
            df[f'{col}_rolling_std_7'] = df.groupby('City')[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).std()
            )
        
        return df
    
    def _handle_outliers(self, df):
        """Handle outliers using IQR method"""
        print("   📊 Handling outliers...")
        
        if len(df) == 0:
            return df
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['City_Encoded', 'Season_Encoded', 'AQI_Bucket_Encoded', 
                       'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 
                       'Quarter', 'IsWeekend', 'Month_sin', 'Month_cos', 
                       'DayOfYear_sin', 'DayOfYear_cos']
        
        outlier_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        for col in outlier_cols:
            if df[col].notna().sum() > 0:  # Only process if we have data
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Only cap outliers if we have variability
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        return df
    
    def _scale_features(self, df):
        """Scale numerical features"""
        print("   ⚖️ Scaling features...")
        
        if len(df) == 0:
            return df
            
        exclude_cols = ['City_Encoded', 'Season_Encoded', 'AQI_Bucket_Encoded', 
                       'AQI', 'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 
                       'WeekOfYear', 'Quarter', 'IsWeekend']
        
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        if numeric_cols:
            self.scalers['feature'] = StandardScaler()
            df[numeric_cols] = self.scalers['feature'].fit_transform(df[numeric_cols])
        
        return df
    def _handle_outliers(self, df):
        """Handle outliers in numerical columns"""
        print("   📊 Handling outliers...")
        
        # Only handle outliers for training, not for prediction
        numerical_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI', 
                        'Temperature', 'Humidity', 'Pressure', 'Wind_Speed']
        
        outlier_count = 0
        for col in numerical_cols:
            if col in df.columns:
                # Use IQR method for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
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
    def _feature_engineering(self, df):
        """Create additional features for better predictions"""
        print("   🔧 Creating additional features...")
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Pollutant ratios and combinations
        if all(col in df.columns for col in ['PM2.5', 'PM10']):
            df['PM_Ratio'] = df['PM2.5'] / df['PM10']
            df['PM_Ratio'] = df['PM_Ratio'].replace([np.inf, -np.inf], np.nan)
            df['PM_Ratio'] = df['PM_Ratio'].fillna(0.5)  # Default ratio
        
        # Air quality indices
        if 'AQI' in df.columns:
            # AQI category buckets
            def get_aqi_category(aqi):
                if pd.isna(aqi):
                    return 'Unknown'
                aqi = float(aqi)
                if aqi <= 50: return 'Good'
                elif aqi <= 100: return 'Satisfactory'
                elif aqi <= 200: return 'Moderate'
                elif aqi <= 300: return 'Poor'
                elif aqi <= 400: return 'Very Poor'
                else: return 'Severe'
            
            df['AQI_Category'] = df['AQI'].apply(get_aqi_category)
            
            # AQI severity score (0-5)
            def get_aqi_severity(aqi):
                if pd.isna(aqi):
                    return 2  # Default to Moderate
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
        
        # Rolling statistics for trends (if we have temporal data)
        if 'Date' in df.columns and 'AQI' in df.columns:
            # Sort by date first
            df = df.sort_values('Date')
            
            # Simple moving averages
            try:
                df['AQI_7Day_Avg'] = df['AQI'].rolling(window=7, min_periods=1).mean()
                df['AQI_30Day_Avg'] = df['AQI'].rolling(window=30, min_periods=1).mean()
            except Exception as e:
                print(f"   ⚠️  Rolling averages failed: {e}")
        
        # City encoding (for ML models)
        if 'City' in df.columns:
            # Create city indicators (one-hot encoding would be done during model training)
            city_mapping = {
                'Delhi': 0, 'Mumbai': 1, 'Bangalore': 2, 'Chennai': 3,
                'Kolkata': 4, 'Hyderabad': 5, 'Ahmedabad': 6, 'Pune': 7
            }
            df['City_Code'] = df['City'].map(city_mapping)
            df['City_Code'] = df['City_Code'].fillna(8)  # Other cities
        
        print("   ✅ Additional features created")
        return df
    def get_feature_columns(self, df, target_col='AQI'):
        """Get list of feature columns for modeling"""
        if len(df) == 0:
            return []
            
        exclude_cols = ['City', 'Date', 'Season', 'AQI_Bucket', target_col, 'Source']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Only return columns that have data (not all NaN)
        valid_feature_cols = []
        for col in feature_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                valid_feature_cols.append(col)
        
        return valid_feature_cols
    
    def prepare_prediction_data(self, historical_data, city, prediction_date):
        """Prepare data for making predictions"""
        # Filter data for the specific city
        city_data = historical_data[historical_data['City'] == city].copy()
        
        if len(city_data) == 0:
            raise ValueError(f"No historical data found for city: {city}")
        
        # Get the latest data point
        latest_data = city_data.sort_values('Date').iloc[-1:].copy()
        
        # Update date to prediction date
        latest_data['Date'] = prediction_date
        latest_data = self._create_temporal_features(latest_data)
        
        return latest_data
    def _final_cleanup(self, df):
        """Final data cleanup and validation"""
        print("   🧹 Final cleanup...")
        
        # Remove any remaining infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN values in numerical columns with median
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Ensure categorical columns have no NaN
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