# Let me reload the full dataset and proceed properly
print("🔄 Reloading Full Synthetic Dataset...")

# Regenerate the complete dataset (same as before but ensuring we have the full version)
np.random.seed(42)

# Define Indian cities
indian_cities = [
    'Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad',
    'Ahmedabad', 'Pune', 'Lucknow', 'Kanpur', 'Patna', 'Gurgaon',
    'Agra', 'Jaipur', 'Coimbatore', 'Surat', 'Indore', 'Bhopal'
]

# Generate full date range
date_range = pd.date_range(start='2015-01-01', end='2020-12-31', freq='D')

# Generate the complete synthetic dataset
def create_full_aqi_dataset():
    data = []
    
    for city in indian_cities:
        city_pollution_base = {
            'Delhi': 200, 'Mumbai': 150, 'Kolkata': 180, 'Chennai': 120,
            'Bangalore': 100, 'Hyderabad': 110, 'Ahmedabad': 140, 'Pune': 90,
            'Lucknow': 170, 'Kanpur': 190, 'Patna': 160, 'Gurgaon': 180,
            'Agra': 150, 'Jaipur': 130, 'Coimbatore': 80, 'Surat': 120,
            'Indore': 110, 'Bhopal': 100
        }
        
        base_pollution = city_pollution_base.get(city, 120)
        
        for date in date_range:
            month = date.month
            seasonal_factor = 1.0
            if month in [12, 1, 2]:  # Winter
                seasonal_factor = 1.8
            elif month in [10, 11]:  # Post-monsoon
                seasonal_factor = 1.5
            elif month in [3, 4]:   # Summer
                seasonal_factor = 1.2
            else:  # Monsoon months
                seasonal_factor = 0.7
            
            weekday_factor = 1.1 if date.weekday() < 5 else 0.85
            random_factor = np.random.uniform(0.5, 1.5)
            
            pm25_base = base_pollution * seasonal_factor * weekday_factor * random_factor * 0.6
            pm10_base = pm25_base * 1.8 + np.random.normal(0, 20)
            
            pm25 = max(10, pm25_base + np.random.normal(0, 15))
            pm10 = max(20, pm10_base + np.random.normal(0, 25))
            no2 = max(5, pm25 * 0.3 + np.random.normal(30, 10))
            so2 = max(2, pm25 * 0.15 + np.random.normal(15, 8))
            co = max(0.1, pm25 * 0.02 + np.random.normal(1.2, 0.4))
            o3 = max(10, np.random.normal(40, 15))
            nh3 = max(5, pm25 * 0.25 + np.random.normal(25, 10))
            benzene = max(1, np.random.normal(8, 3))
            toluene = max(1, np.random.normal(12, 4))
            xylene = max(1, np.random.normal(6, 2))
            
            aqi = min(500, max(0, pm25 * 4.17))
            
            if aqi <= 50:
                aqi_bucket = "Good"
            elif aqi <= 100:
                aqi_bucket = "Satisfactory"
            elif aqi <= 200:
                aqi_bucket = "Moderate"
            elif aqi <= 300:
                aqi_bucket = "Poor"
            elif aqi <= 400:
                aqi_bucket = "Very Poor"
            else:
                aqi_bucket = "Severe"
            
            data.append({
                'City': city,
                'Date': date,
                'PM2.5': round(pm25, 2),
                'PM10': round(pm10, 2),
                'NO': round(no2 * 0.7, 2),
                'NO2': round(no2, 2),
                'NH3': round(nh3, 2),
                'CO': round(co, 2),
                'SO2': round(so2, 2),
                'O3': round(o3, 2),
                'Benzene': round(benzene, 2),
                'Toluene': round(toluene, 2),
                'Xylene': round(xylene, 2),
                'AQI': round(aqi, 0),
                'AQI_Bucket': aqi_bucket
            })
    
    return pd.DataFrame(data)

# Create the full dataset
full_df = create_full_aqi_dataset()

# Add time features
full_df['Year'] = full_df['Date'].dt.year
full_df['Month'] = full_df['Date'].dt.month
full_df['Day'] = full_df['Date'].dt.day
full_df['DayOfWeek'] = full_df['Date'].dt.dayofweek
full_df['Season'] = full_df['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})

# Introduce some missing values
np.random.seed(42)
missing_indices = np.random.choice(full_df.index, size=int(0.03 * len(full_df)), replace=False)
full_df.loc[missing_indices[:len(missing_indices)//3], 'NH3'] = np.nan
full_df.loc[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3], 'Benzene'] = np.nan
full_df.loc[missing_indices[2*len(missing_indices)//3:], 'SO2'] = np.nan

print(f"✅ Full dataset recreated:")
print(f"   Shape: {full_df.shape}")
print(f"   Date range: {full_df['Date'].min().date()} to {full_df['Date'].max().date()}")
print(f"   Cities: {len(full_df['City'].unique())} unique cities")
print(f"   Missing values: {full_df.isnull().sum().sum()} total")

# Now preprocess this full dataset
df_final, feature_cols, pollutant_cols = preprocess_aqi_data(full_df)
print(f"\n🎯 Final dataset ready for modeling!")
print(f"   Shape: {df_final.shape}")
print(f"   Features: {len(feature_cols)} columns")
print(f"   Pollutants: {pollutant_cols}")

# Save the complete processed dataset
df_final.to_csv('india_aqi_complete_processed.csv', index=False)
print(f"💾 Complete dataset saved!")

print(f"\n📊 Quick stats:")
print(f"   Total records: {len(df_final):,}")
print(f"   AQI range: {df_final['AQI'].min():.1f} - {df_final['AQI'].max():.1f}")
print(f"   Missing values remaining: {df_final.isnull().sum().sum()}")

# Display sample
print(f"\nSample of complete processed data:")
print(df_final[['City', 'Date', 'PM2.5', 'PM10', 'AQI', 'Season', 'City_Encoded']].head())