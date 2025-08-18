# Simple and fast model training
print("🤖 Quick Model Training & Results")

# Use a very small sample for demonstration
np.random.seed(42)
sample_data = df_final.sample(n=1000, random_state=42)

# Quick model training
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Simple features
X_simple = sample_data[['PM2.5', 'PM10', 'NO2', 'City_Encoded', 'Month']]
y_simple = sample_data['AQI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_simple, y_simple, test_size=0.3, random_state=42)

# Quick models
models_quick = {
    'Random Forest': RandomForestRegressor(n_estimators=10, random_state=42),
    'Linear Regression': LinearRegression()
}

# Train and get results
results_summary = {}
for name, model in models_quick.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results_summary[name] = {'RMSE': rmse, 'R2': r2}
    print(f"{name}: RMSE={rmse:.1f}, R²={r2:.3f}")

print("\n✅ Quick training completed!")
print("📊 Sample predictions:", y_pred[:5])
print("🎯 Ready for dashboard development!")

# Create summary data for dashboard
summary_stats = {
    'total_records': len(df_final),
    'cities_count': df_final['City'].nunique(),
    'date_range': f"{df_final['Date'].min().date()} to {df_final['Date'].max().date()}",
    'avg_aqi': df_final['AQI'].mean(),
    'max_aqi': df_final['AQI'].max(),
    'best_model': 'Random Forest',
    'model_accuracy': results_summary['Random Forest']['R2']
}

print(f"\n📈 Dataset Summary for Dashboard:")
for key, value in summary_stats.items():
    print(f"  {key}: {value}")

# Save summary for dashboard
import json
with open('project_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2, default=str)
    
print(f"\n💾 Project summary saved!")
print(f"🚀 Ready to build dashboard!")