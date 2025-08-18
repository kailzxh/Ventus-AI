# Step 4: Model Training and Evaluation
print("🤖 Step 4: Machine Learning Model Training")
print("="*50)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Prepare data for modeling
print("📊 Preparing Data for Machine Learning...")

# Define features and target
X_features = [col for col in feature_cols if col != 'AQI_Bucket_Encoded']  # Remove AQI bucket since it's derived from AQI
y_target = 'AQI'

print(f"Features to use: {X_features}")
print(f"Target variable: {y_target}")

# Prepare feature matrix and target vector
X = df_final[X_features].values
y = df_final[y_target].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Train/Test split with temporal consideration
print("\n🔄 Creating Train/Test Splits...")

# Sort by date to ensure temporal order
df_sorted = df_final.sort_values('Date')
split_date = pd.Timestamp('2019-01-01')  # Use 2019+ as test set

train_mask = df_sorted['Date'] < split_date
test_mask = df_sorted['Date'] >= split_date

X_train = df_sorted[train_mask][X_features].values
X_test = df_sorted[test_mask][X_features].values
y_train = df_sorted[train_mask][y_target].values
y_test = df_sorted[test_mask][y_target].values

print(f"Training set: {X_train.shape[0]} samples ({train_mask.sum()})")
print(f"Test set: {X_test.shape[0]} samples ({test_mask.sum()})")
print(f"Split date: {split_date.date()}")

# Feature scaling
print("\n🔧 Applying Feature Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling completed")

# Define models to train (following the baseline models from the paper)
print("\n🎯 Defining Baseline Models...")

models = {
    'Linear_Regression': LinearRegression(),
    'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Neural_Network': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    'Support_Vector_Regression': SVR(kernel='rbf')
}

print(f"Models to train: {list(models.keys())}")

# Train and evaluate models
print("\n🚀 Training Models...")
results = {}

for name, model in models.items():
    print(f"\n  📈 Training {name}...")
    
    try:
        # Use scaled features for neural networks and SVR, raw features for tree-based models
        if name in ['Neural_Network', 'Support_Vector_Regression']:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        # Train model
        model.fit(X_train_use, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_use)
        y_pred_test = model.predict(X_test_use)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Store results
        results[name] = {
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse,
            'Train_MAE': train_mae,
            'Test_MAE': test_mae,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Model': model
        }
        
        print(f"     ✅ {name} completed")
        print(f"        Test RMSE: {test_rmse:.2f}")
        print(f"        Test MAE: {test_mae:.2f}")
        print(f"        Test R²: {test_r2:.3f}")
        
    except Exception as e:
        print(f"     ❌ {name} failed: {str(e)}")
        continue

print(f"\n✅ Model training completed! {len(results)} models trained successfully.")

# Create results summary
print("\n📊 Model Performance Summary:")
print("="*80)
results_df = pd.DataFrame(results).T
print(results_df[['Test_RMSE', 'Test_MAE', 'Test_R2']].round(3))

# Save results
results_df.to_csv('model_results.csv')
print(f"\n💾 Results saved to 'model_results.csv'")

# Save the best model
best_model_name = results_df['Test_R2'].idxmax()
best_model = results[best_model_name]['Model']
joblib.dump(best_model, f'best_model_{best_model_name.lower()}.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test R²: {results[best_model_name]['Test_R2']:.3f}")
print(f"   Test RMSE: {results[best_model_name]['Test_RMSE']:.2f}")
print(f"   Test MAE: {results[best_model_name]['Test_MAE']:.2f}")
print(f"   Saved as: best_model_{best_model_name.lower()}.pkl")

print(f"\n🎯 Models ready for deployment and dashboard integration!")