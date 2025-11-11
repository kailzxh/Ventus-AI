// test-apis.js
const API_BASE = 'http://localhost:5000/api';

async function testAPI(endpoint, options = {}) {
  try {
    const url = `${API_BASE}${endpoint}`;
    console.log(`🔍 Testing: ${url}`);
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }
    
    const data = await response.json();
    console.log('✅ Success');
    return data;
  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
    return null;
  }
}

// 🎯 SPECIFIC MODEL TESTING APIS

async function testAllModelsForCity(city, date = '2024-01-15') {
  console.log(`\n🎯 Testing ALL Models for ${city} on ${date}`);
  
  const models = ['auto', 'nf_vae', 'random_forest', 'gradient_boosting', 'simple'];
  const results = [];
  
  for (const model of models) {
    console.log(`\n🤖 Testing ${model.toUpperCase()} model...`);
    const result = await testAPI('/predict', {
      method: 'POST',
      body: JSON.stringify({
        city: city,
        date: date,
        model_type: model
      })
    });
    
    if (result) {
      results.push({
        model_requested: model,
        model_used: result.model_used,
        predicted_aqi: result.predicted_aqi,
        category: result.category,
        confidence: result.confidence,
        model_loaded: result.model_loaded,
        source: result.source,
        timestamp: result.timestamp
      });
      console.log(`   📊 Requested: ${model} | Used: ${result.model_used}`);
      console.log(`   🎯 AQI: ${result.predicted_aqi} | Category: ${result.category}`);
      console.log(`   ✅ Confidence: ${result.confidence} | Loaded: ${result.model_loaded}`);
    }
  }
  
  return results;
}

async function testModelPerformance() {
  console.log('\n📈 Testing Model Performance Metrics');
  
  const performance = await testAPI('/models/performance');
  if (performance) {
    console.log('📊 Model Performance Metrics:');
    Object.entries(performance.model_performance || performance).forEach(([model, metrics]) => {
      console.log(`   ${model.toUpperCase()}: RMSE=${metrics.RMSE}, MAE=${metrics.MAE}, R2=${metrics.R2}`);
    });
  }
  
  return performance;
}

async function testAvailableModels() {
  console.log('\n🛠️ Testing Available Models');
  
  const models = await testAPI('/models/available');
  if (models) {
    console.log('🤖 Available Models:');
    console.log(`   Default: ${models.default_model}`);
    console.log(`   Loaded: ${models.available_models.join(', ')}`);
    console.log('   Descriptions:');
    Object.entries(models.model_descriptions || {}).forEach(([model, desc]) => {
      console.log(`     ${model}: ${desc}`);
    });
  }
  
  return models;
}

async function testNFVAEModelSpecific(city = 'Delhi') {
  console.log(`\n🧠 Testing NF-VAE Model Specific Features for ${city}`);
  
  const tests = [
    {
      name: 'NF-VAE Single Prediction',
      endpoint: '/predict',
      body: { city: city, date: '2024-01-15', model_type: 'nf_vae' }
    },
    {
      name: 'NF-VAE Future Predictions',
      endpoint: `/predict/${city}/future?days=7&model_type=nf_vae`,
      method: 'GET'
    },
    {
      name: 'NF-VAE Station Predictions',
      endpoint: '/predict/stations',
      body: { city: city, date: '2024-01-15', model_type: 'nf_vae' }
    }
  ];
  
  const results = [];
  
  for (const test of tests) {
    console.log(`\n🔬 ${test.name}...`);
    const result = await testAPI(test.endpoint, {
      method: test.method || 'POST',
      body: test.body ? JSON.stringify(test.body) : undefined
    });
    
    if (result) {
      results.push({
        test: test.name,
        success: true,
        data: test.name.includes('Future') ? `${result.predictions?.length || 0} predictions` : 
              test.name.includes('Station') ? `${result.total_predictions || 0} station predictions` :
              `AQI: ${result.predicted_aqi}`
      });
    }
  }
  
  return results;
}

async function testRandomForestModelSpecific(city = 'Delhi') {
  console.log(`\n🌳 Testing Random Forest Model Specific Features for ${city}`);
  
  const tests = [
    {
      name: 'Random Forest Single Prediction',
      endpoint: '/predict',
      body: { city: city, date: '2024-01-15', model_type: 'random_forest' }
    },
    {
      name: 'Random Forest Feature Importance',
      endpoint: '/predict',
      body: { city: city, date: '2024-01-15', model_type: 'random_forest', include_features: true }
    }
  ];
  
  const results = [];
  
  for (const test of tests) {
    console.log(`\n🔬 ${test.name}...`);
    const result = await testAPI(test.endpoint, {
      method: 'POST',
      body: JSON.stringify(test.body)
    });
    
    if (result) {
      results.push({
        test: test.name,
        model_used: result.model_used,
        predicted_aqi: result.predicted_aqi,
        confidence: result.confidence,
        model_loaded: result.model_loaded
      });
      console.log(`   📊 Used: ${result.model_used} | AQI: ${result.predicted_aqi}`);
      console.log(`   ✅ Loaded: ${result.model_loaded} | Confidence: ${result.confidence}`);
    }
  }
  
  return results;
}

async function testGradientBoostingModelSpecific(city = 'Delhi') {
  console.log(`\n🚀 Testing Gradient Boosting Model Specific Features for ${city}`);
  
  const tests = [
    {
      name: 'Gradient Boosting Single Prediction',
      endpoint: '/predict',
      body: { city: city, date: '2024-01-15', model_type: 'gradient_boosting' }
    },
    {
      name: 'Gradient Boosting Multiple Cities',
      endpoint: '/predict',
      body: { city: city, date: '2024-01-15', model_type: 'gradient_boosting', compare_cities: true }
    }
  ];
  
  const results = [];
  
  for (const test of tests) {
    console.log(`\n🔬 ${test.name}...`);
    const result = await testAPI(test.endpoint, {
      method: 'POST',
      body: JSON.stringify(test.body)
    });
    
    if (result) {
      results.push({
        test: test.name,
        model_used: result.model_used,
        predicted_aqi: result.predicted_aqi,
        confidence: result.confidence,
        model_loaded: result.model_loaded
      });
      console.log(`   📊 Used: ${result.model_used} | AQI: ${result.predicted_aqi}`);
      console.log(`   ✅ Loaded: ${result.model_loaded} | Confidence: ${result.confidence}`);
    }
  }
  
  return results;
}

async function testSimpleModelSpecific(city = 'Delhi') {
  console.log(`\n📊 Testing Simple Model Specific Features for ${city}`);
  
  const tests = [
    {
      name: 'Simple Model Single Prediction',
      endpoint: '/predict',
      body: { city: city, date: '2024-01-15', model_type: 'simple' }
    },
    {
      name: 'Simple Model Historical Pattern',
      endpoint: '/predict',
      body: { city: city, date: '2024-01-15', model_type: 'simple', include_pattern: true }
    }
  ];
  
  const results = [];
  
  for (const test of tests) {
    console.log(`\n🔬 ${test.name}...`);
    const result = await testAPI(test.endpoint, {
      method: 'POST',
      body: JSON.stringify(test.body)
    });
    
    if (result) {
      results.push({
        test: test.name,
        model_used: result.model_used,
        predicted_aqi: result.predicted_aqi,
        confidence: result.confidence,
        source: result.source
      });
      console.log(`   📊 Used: ${result.model_used} | AQI: ${result.predicted_aqi}`);
      console.log(`   ✅ Source: ${result.source} | Confidence: ${result.confidence}`);
    }
  }
  
  return results;
}

async function testModelComparison(cities = ['Delhi', 'Mumbai', 'Bangalore']) {
  console.log('\n📊 Testing Model Comparison Across Cities');
  
  const comparisonResults = {};
  
  for (const city of cities) {
    console.log(`\n🏙️ Comparing models for ${city}:`);
    comparisonResults[city] = await testAllModelsForCity(city);
  }
  
  // Create comparison table
  console.log('\n📈 MODEL COMPARISON SUMMARY');
  console.log('=' .repeat(100));
  console.log('City'.padEnd(15) + 'Model'.padEnd(20) + 'AQI'.padEnd(10) + 'Category'.padEnd(20) + 'Loaded'.padEnd(10) + 'Confidence');
  console.log('-'.repeat(100));
  
  for (const [city, results] of Object.entries(comparisonResults)) {
    results.forEach(result => {
      console.log(
        city.padEnd(15) +
        result.model_used.padEnd(20) +
        String(result.predicted_aqi).padEnd(10) +
        result.category.padEnd(20) +
        String(result.model_loaded).padEnd(10) +
        result.confidence
      );
    });
    console.log('-'.repeat(100));
  }
  
  return comparisonResults;
}

async function testRealTimeModelComparison() {
  console.log('\n⚡ Testing Real-time vs Model Predictions');
  
  const cities = ['Delhi', 'Mumbai', 'Bangalore'];
  const results = [];
  
  for (const city of cities) {
    console.log(`\n🔍 Real-time comparison for ${city}:`);
    
    const realtimeData = await testAPI(`/realtime/${city}`);
    if (realtimeData) {
      const currentAQI = realtimeData.realtime?.aqi;
      const predictions = realtimeData.predictions || {};
      
      console.log(`   📊 Current AQI: ${currentAQI} (${realtimeData.realtime?.category})`);
      console.log(`   🔮 Today's Prediction: ${predictions.today?.predicted_aqi} (${realtimeData.accuracy?.today}% accuracy)`);
      console.log(`   🤖 Model Used: ${predictions.today?.model_used || 'Unknown'}`);
      console.log(`   📈 Trend: ${realtimeData.trend_analysis?.short_term}`);
      
      results.push({
        city: city,
        current_aqi: currentAQI,
        predicted_aqi: predictions.today?.predicted_aqi,
        accuracy: realtimeData.accuracy?.today,
        model_used: predictions.today?.model_used,
        trend: realtimeData.trend_analysis?.short_term
      });
    }
  }
  
  return results;
}

async function testModelFallbackScenarios() {
  console.log('\n🔄 Testing Model Fallback Scenarios');
  
  const fallbackTests = [
    {
      name: 'Invalid city with auto model',
      body: { city: 'InvalidCity123', date: '2024-01-15', model_type: 'auto' },
      expected: 'Should use simple fallback'
    },
    {
      name: 'Valid city with invalid model',
      body: { city: 'Delhi', date: '2024-01-15', model_type: 'invalid_model' },
      expected: 'Should fallback to default model'
    },
    {
      name: 'Distant future date',
      body: { city: 'Delhi', date: '2025-12-31', model_type: 'nf_vae' },
      expected: 'Should handle long-term prediction'
    }
  ];
  
  const results = [];
  
  for (const test of fallbackTests) {
    console.log(`\n🔄 Testing: ${test.name}`);
    console.log(`   Expected: ${test.expected}`);
    
    const result = await testAPI('/predict', {
      method: 'POST',
      body: JSON.stringify(test.body)
    });
    
    if (result) {
      results.push({
        test: test.name,
        model_requested: test.body.model_type,
        model_used: result.model_used,
        predicted_aqi: result.predicted_aqi,
        success: result.model_used !== test.body.model_type ? 'Fallback used' : 'Original model used'
      });
      console.log(`   ✅ Result: Used ${result.model_used} | AQI: ${result.predicted_aqi}`);
    }
  }
  
  return results;
}

// 🚀 COMPREHENSIVE TEST SUITE

async function runModelSpecificTests() {
  console.log('🚀 Starting Model-Specific API Tests...\n');
  
  try {
    // 1. System Status
    console.log('📋 1. SYSTEM STATUS AND MODELS');
    await testAPI('/health');
    await testAPI('/status');
    await testAvailableModels();
    await testModelPerformance();
    
    // 2. Test Each Model Individually
    console.log('\n🤖 2. INDIVIDUAL MODEL TESTING');
    
    console.log('\n🧠 2.1 NF-VAE MODEL TESTS');
    await testNFVAEModelSpecific('Delhi');
    await testNFVAEModelSpecific('Mumbai');
    
    console.log('\n🌳 2.2 RANDOM FOREST MODEL TESTS');
    await testRandomForestModelSpecific('Delhi');
    await testRandomForestModelSpecific('Bangalore');
    
    console.log('\n🚀 2.3 GRADIENT BOOSTING MODEL TESTS');
    await testGradientBoostingModelSpecific('Delhi');
    await testGradientBoostingModelSpecific('Chennai');
    
    console.log('\n📊 2.4 SIMPLE MODEL TESTS');
    await testSimpleModelSpecific('Delhi');
    await testSimpleModelSpecific('Kolkata');
    
    // 3. Model Comparison
    console.log('\n📈 3. MODEL COMPARISON TESTS');
    await testModelComparison(['Delhi', 'Mumbai', 'Bangalore', 'Chennai']);
    
    // 4. Real-time Comparisons
    console.log('\n⚡ 4. REAL-TIME VS MODEL PREDICTIONS');
    await testRealTimeModelComparison();
    
    // 5. Fallback Scenarios
    console.log('\n🔄 5. FALLBACK SCENARIO TESTS');
    await testModelFallbackScenarios();
    
    // 6. Comprehensive Model Testing for All Cities
    console.log('\n🏙️ 6. COMPREHENSIVE CITY MODEL TESTING');
    const majorCities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad'];
    
    for (const city of majorCities) {
      console.log(`\n🎯 Testing all models for ${city}:`);
      await testAllModelsForCity(city);
    }
    
    console.log('\n🎉 ALL MODEL-SPECIFIC TESTS COMPLETED!');
    
  } catch (error) {
    console.error('\n💥 MODEL TEST SUITE FAILED:', error);
  }
}

// 🎯 QUICK TEST FUNCTIONS

async function quickModelTest(city = 'Delhi') {
  console.log(`🚀 Quick Model Test for ${city}`);
  return await testAllModelsForCity(city);
}

async function quickSystemCheck() {
  console.log('🔍 Quick System Check');
  const health = await testAPI('/health');
  const models = await testAvailableModels();
  const performance = await testModelPerformance();
  
  return { health, models, performance };
}

// Run the comprehensive tests
if (require.main === module) {
  runModelSpecificTests();
}

// Export all test functions
module.exports = {
  // Core testing
  testAPI,
  
  // Model-specific tests
  testAllModelsForCity,
  testNFVAEModelSpecific,
  testRandomForestModelSpecific,
  testGradientBoostingModelSpecific,
  testSimpleModelSpecific,
  
  // Comparison tests
  testModelComparison,
  testRealTimeModelComparison,
  testModelFallbackScenarios,
  
  // System tests
  testAvailableModels,
  testModelPerformance,
  
  // Quick tests
  quickModelTest,
  quickSystemCheck,
  
  // Comprehensive suite
  runModelSpecificTests
};