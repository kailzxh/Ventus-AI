# Step 2: Data Exploration and Analysis
print("🔍 Step 2: Exploratory Data Analysis")
print("="*50)

# Basic statistics
print("📊 Dataset Statistics:")
print(df.describe())

print(f"\n🏙️ Cities in dataset: {list(df['City'].unique())}")
print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"⏱️ Total days: {(df['Date'].max() - df['Date'].min()).days} days")

# Check for missing values
print("\n🔍 Missing Values Check:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found! ✅")

# AQI distribution by bucket
print("\n📈 AQI Distribution by Category:")
aqi_distribution = df['AQI_Bucket'].value_counts()
print(aqi_distribution)

# Add some realistic missing values to simulate real-world data
print("\n🔧 Introducing realistic missing values for preprocessing practice...")

# Randomly introduce missing values in some columns (common in real AQI data)
np.random.seed(42)
missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
df.loc[missing_indices[:len(missing_indices)//3], 'NH3'] = np.nan
df.loc[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3], 'Benzene'] = np.nan
df.loc[missing_indices[2*len(missing_indices)//3:], 'Toluene'] = np.nan

# Add a few missing values to other columns
missing_indices_2 = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
df.loc[missing_indices_2[:len(missing_indices_2)//2], 'SO2'] = np.nan
df.loc[missing_indices_2[len(missing_indices_2)//2:], 'O3'] = np.nan

print("✅ Missing values introduced to simulate real-world conditions")
print("\n🔍 Updated Missing Values:")
missing_values_updated = df.isnull().sum()
print(missing_values_updated[missing_values_updated > 0])

# Save the raw dataset
print("\n💾 Saving raw dataset...")
df.to_csv('india_aqi_raw_data.csv', index=False)
print("✅ Raw dataset saved as 'india_aqi_raw_data.csv'")

# Create additional time-based features for better prediction
print("\n⏰ Creating time-based features...")
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Season'] = df['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})

print("✅ Time-based features created: Year, Month, Day, DayOfWeek, Season")
print(f"Updated dataset shape: {df.shape}")

# Display sample with new features
print("\n📊 Sample data with new features:")
print(df[['City', 'Date', 'PM2.5', 'PM10', 'AQI', 'Year', 'Month', 'Season', 'DayOfWeek']].head())