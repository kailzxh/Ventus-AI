# Streamlined Model Training - Focus on key models for efficiency
print("🤖 Step 4: Streamlined Machine Learning Model Training")
print("="*55)

# Quick data preparation
print("📊 Preparing Data for ML (Streamlined)...")

# Use a subset for faster training
df_sample = df_final.sample(n=5000, random_state=42).copy()  # Use 5k samples for speed
df_sample = df_sample.sort_values('Date')

# Define features and target
feature_list = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'City_Encoded', 'Season_Encoded', 'Month']
X = df_sample[feature_list]
y = df_sample['AQI']

print(f"Sample dataset shape: {df_sample.shape}")
print(f"Features: {feature_list}")
print(f"Feature matrix: {X.shape}, Target: {y.shape}")

# Simple train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define streamlined models
models = {
    'Random_Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    'Gradient_Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
    'Linear_Regression': LinearRegression()
}

print(f"\n🚀 Training {len(models)} models...")

# Train models quickly
results = {}
for name, model in models.items():
    print(f"  📈 Training {name}...")
    
    # Use appropriate data
    if name == 'Linear_Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Model': model
    }
    
    print(f"     RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

print(f"\n✅ Training completed!")

# Results summary
print(f"\n📊 Model Performance Comparison:")
print("="*50)
for name, metrics in results.items():
    print(f"{name:20} | RMSE: {metrics['RMSE']:6.2f} | MAE: {metrics['MAE']:6.2f} | R²: {metrics['R2']:6.3f}")

# Best model
best_model = max(results.items(), key=lambda x: x[1]['R2'])
print(f"\n🏆 Best Model: {best_model[0]} (R² = {best_model[1]['R2']:.3f})")

# Save models and results
import pickle
with open('trained_models.pkl', 'wb') as f:
    pickle.dump(results, f)
    
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"💾 Models and scaler saved!")

# Feature importance (for Random Forest)
if 'Random_Forest' in results:
    rf_model = results['Random_Forest']['Model']
    feature_importance = pd.DataFrame({
        'Feature': feature_list,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🔍 Feature Importance (Random Forest):")
    print(feature_importance)

print(f"\n🎯 Models trained and ready for dashboard integration!")
print(f"Next: Dashboard development and deployment")