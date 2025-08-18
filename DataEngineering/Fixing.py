# Let's check the actual column names and fix the preprocessing
print("🔍 Checking actual column names in the dataset...")
print("Available columns:", list(df.columns))
print("\nMissing values by column:")
print(df.isnull().sum())

# Fix the preprocessing function with correct column names
def preprocess_data_fixed(df_input):
    df = df_input.copy()
    
    print("🧹 Handling Missing Values...")
    
    # Get columns that actually have missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    print(f"Columns with missing values: {missing_cols}")
    
    # Strategy: Use median for numeric columns (right-skewed distributions)
    for col in missing_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            missing_count = df[col].isnull().sum()
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"  - {col}: Filled {missing_count} missing values with median ({median_value:.2f})")
    
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

# Preprocess the data with correct function
df_processed, city_encoder, season_encoder = preprocess_data_fixed(df)

print(f"\n📊 Processed dataset shape: {df_processed.shape}")
print("✅ All preprocessing steps completed")

# Create feature sets for different prediction tasks
print("\n🎯 Creating Feature Sets for Model Training...")

# Define feature columns based on actual columns in the dataset
available_numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
pollutant_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Filter to only include columns that exist
pollutant_features = [col for col in pollutant_features if col in df_processed.columns]
temporal_features = ['Year', 'Month', 'Day', 'DayOfWeek']
spatial_features = ['City_Encoded']
seasonal_features = ['Season_Encoded']

# All features combined
all_features = pollutant_features + temporal_features + spatial_features + seasonal_features

print(f"📋 Feature groups defined:")
print(f"  - Pollutant features: {pollutant_features}")
print(f"  - Temporal features: {temporal_features}")
print(f"  - Spatial features: {spatial_features}")
print(f"  - Seasonal features: {seasonal_features}")
print(f"  - Total features: {len(all_features)} features")

# Display correlation matrix of key pollutants
print("\n🔍 Analyzing Feature Correlations...")
corr_features = [col for col in pollutant_features if col in df_processed.columns] + ['AQI']
correlation_matrix = df_processed[corr_features].corr()
print("Top correlations with AQI:")
aqi_correlations = correlation_matrix['AQI'].abs().sort_values(ascending=False)
for feature, corr in aqi_correlations[1:6].items():  # Top 5 excluding AQI itself
    print(f"  - {feature}: {corr:.3f}")

print(f"\n✅ Data preprocessing completed successfully!")
print(f"Final processed dataset shape: {df_processed.shape}")
print(f"No missing values remaining: {df_processed.isnull().sum().sum() == 0}")