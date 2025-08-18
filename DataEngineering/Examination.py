# Let's first examine the dataframe structure properly
print("🔍 Complete Dataset Structure Analysis:")
print("="*50)
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame columns: {list(df.columns)}")
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

# Let's create a simplified but robust preprocessing function
def preprocess_aqi_data(df_input):
    """
    Robust preprocessing function for AQI data
    """
    df = df_input.copy()
    
    print("🧹 Starting Data Preprocessing...")
    
    # 1. Handle missing values
    print("\n1️⃣ Handling Missing Values:")
    missing_before = df.isnull().sum().sum()
    print(f"   Total missing values before: {missing_before}")
    
    # Fill missing values for numeric columns using median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            missing_count = df[col].isnull().sum()
            if col != 'AQI':  # Don't fill target variable
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"   - Filled {missing_count} values in {col} with median: {median_val:.2f}")
    
    missing_after = df.isnull().sum().sum()
    print(f"   Total missing values after: {missing_after}")
    
    # 2. Encode categorical variables
    print("\n2️⃣ Encoding Categorical Variables:")
    
    # City encoding
    if 'City' in df.columns:
        le_city = LabelEncoder()
        df['City_Encoded'] = le_city.fit_transform(df['City'])
        print(f"   - City encoded: {len(le_city.classes_)} unique cities")
    
    # Season encoding
    if 'Season' in df.columns:
        le_season = LabelEncoder()
        df['Season_Encoded'] = le_season.fit_transform(df['Season'])
        print(f"   - Season encoded: {le_season.classes_}")
    
    # AQI_Bucket encoding (if exists)
    if 'AQI_Bucket' in df.columns:
        le_aqi = LabelEncoder()
        df['AQI_Bucket_Encoded'] = le_aqi.fit_transform(df['AQI_Bucket'])
        print(f"   - AQI_Bucket encoded: {le_aqi.classes_}")
    
    # 3. Feature scaling preparation
    print("\n3️⃣ Preparing features for scaling...")
    
    # Define feature groups
    pollutant_cols = [col for col in ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3', 
                                     'Benzene', 'Toluene', 'Xylene'] if col in df.columns]
    
    temporal_cols = [col for col in ['Year', 'Month', 'Day', 'DayOfWeek'] if col in df.columns]
    
    encoded_cols = [col for col in df.columns if '_Encoded' in col]
    
    feature_cols = pollutant_cols + temporal_cols + encoded_cols
    
    print(f"   - Pollutant features: {pollutant_cols}")
    print(f"   - Temporal features: {temporal_cols}")
    print(f"   - Encoded features: {encoded_cols}")
    print(f"   - Total feature columns: {len(feature_cols)}")
    
    return df, feature_cols, pollutant_cols

# Apply preprocessing
df_processed, feature_columns, pollutant_columns = preprocess_aqi_data(df)

print(f"\n✅ Preprocessing completed!")
print(f"Processed dataset shape: {df_processed.shape}")
print(f"Available features: {len(feature_columns)} columns")

# Save the preprocessed dataset
df_processed.to_csv('india_aqi_preprocessed.csv', index=False)
print(f"💾 Saved preprocessed dataset to 'india_aqi_preprocessed.csv'")

# Display sample of processed data
print(f"\n📊 Sample of preprocessed data:")
sample_cols = ['City', 'Date', 'PM2.5', 'AQI', 'City_Encoded', 'Season_Encoded'][:6]
available_sample_cols = [col for col in sample_cols if col in df_processed.columns]
print(df_processed[available_sample_cols].head())

print(f"\n🎯 Ready for Model Training!")
print(f"Target variable: AQI")
print(f"Feature variables: {len(feature_columns)} features")