# Step 3: Data Preprocessing for ML Models
print("🔧 Step 3: Data Preprocessing for Machine Learning")
print("="*55)

# Handle missing values using different strategies (following the paper's approach)
def preprocess_data(df_input):
    df = df_input.copy()
    
    print("🧹 Handling Missing Values...")
    
    # Strategy 1: For NH3, Benzene, Toluene - use median (right-skewed distributions)
    numeric_cols = ['NH3', 'Benzene', 'Toluene', 'SO2', 'O3']
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            # Use median for right-skewed pollutant data
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"  - {col}: Filled {df_input[col].isnull().sum()} missing values with median ({median_value:.2f})")
    
    print("✅ Missing value imputation completed")
    
    # Encode categorical variables
    print("\n🏷️ Encoding Categorical Variables...")
    
    # Label encoding for City
    le_city = LabelEncoder()
    df['City_Encoded'] = le_city.fit_transform(df['City'])
    
    # Label encoding for Season
    le_season = LabelEncoder()
    df['Season_Encoded'] = le_season.fit_transform(df['Season'])
    
    # One-hot encoding for AQI_Bucket (if needed for classification)
    aqi_dummies = pd.get_dummies(df['AQI_Bucket'], prefix='AQI_Bucket')
    df = pd.concat([df, aqi_dummies], axis=1)
    
    print("✅ Categorical encoding completed")
    
    return df, le_city, le_season

# Preprocess the data
df_processed, city_encoder, season_encoder = preprocess_data(df)

print(f"\n📊 Processed dataset shape: {df_processed.shape}")
print("✅ All preprocessing steps completed")

# Create feature sets for different prediction tasks
print("\n🎯 Creating Feature Sets for Model Training...")

# Define feature columns (following NF-VAE paper approach)
pollutant_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
temporal_features = ['Year', 'Month', 'Day', 'DayOfWeek']
spatial_features = ['City_Encoded']
seasonal_features = ['Season_Encoded']

# All features combined
all_features = pollutant_features + temporal_features + spatial_features + seasonal_features

print(f"📋 Feature groups defined:")
print(f"  - Pollutant features: {len(pollutant_features)} features")
print(f"  - Temporal features: {len(temporal_features)} features")
print(f"  - Spatial features: {len(spatial_features)} features")
print(f"  - Seasonal features: {len(seasonal_features)} features")
print(f"  - Total features: {len(all_features)} features")

# Display correlation matrix of key pollutants
print("\n🔍 Analyzing Feature Correlations...")
correlation_matrix = df_processed[pollutant_features + ['AQI']].corr()
print("Top correlations with AQI:")
aqi_correlations = correlation_matrix['AQI'].abs().sort_values(ascending=False)
for feature, corr in aqi_correlations[1:6].items():  # Top 5 excluding AQI itself
    print(f"  - {feature}: {corr:.3f}")

# Prepare data for time series modeling (following paper's approach)
print("\n⏰ Preparing Time Series Data...")

# Sort by City and Date for proper time series order
df_processed = df_processed.sort_values(['City', 'Date'])

# Create lagged features (important for time series prediction)
def create_lag_features(group_df, feature_cols, lags=[1, 2, 3, 7]):
    """Create lagged features for time series modeling"""
    for col in feature_cols:
        for lag in lags:
            group_df[f'{col}_lag_{lag}'] = group_df[col].shift(lag)
    return group_df

# Apply lag features by city
print("Creating lagged features for time series modeling...")
key_pollutants = ['PM2.5', 'PM10', 'NO2', 'AQI']  # Focus on key pollutants for lags
df_with_lags = df_processed.groupby('City').apply(
    lambda x: create_lag_features(x, key_pollutants, lags=[1, 3, 7])
).reset_index(drop=True)

print(f"✅ Lagged features created. New shape: {df_with_lags.shape}")

# Handle NaN values created by lagging (drop first few rows per city)
lag_features = [col for col in df_with_lags.columns if '_lag_' in col]
df_with_lags = df_with_lags.dropna(subset=lag_features)

print(f"📊 Final dataset after lag processing: {df_with_lags.shape}")

# Save processed dataset
print("\n💾 Saving processed dataset...")
df_with_lags.to_csv('india_aqi_processed_data.csv', index=False)
print("✅ Processed dataset saved as 'india_aqi_processed_data.csv'")

# Display sample of processed data
print("\n📊 Sample of processed data with new features:")
sample_cols = ['City', 'Date', 'PM2.5', 'PM2.5_lag_1', 'PM2.5_lag_7', 'AQI', 'City_Encoded', 'Season_Encoded']
print(df_with_lags[sample_cols].head(10))