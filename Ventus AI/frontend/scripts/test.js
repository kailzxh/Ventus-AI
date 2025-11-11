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

async function testModelPredictions(city) {
  console.log(`\n🎯 Testing Model Predictions for ${city}`);
  
  const models = ['auto', 'nf_vae', 'random_forest', 'gradient_boosting'];
  const results = [];
  
  for (const model of models) {
    console.log(`\n🤖 Testing ${model} model...`);
    const result = await testAPI('/predict', {
      method: 'POST',
      body: JSON.stringify({
        city: city,
        date: '2024-01-15',
        model_type: model
      })
    });
    
    if (result) {
      results.push({
        model,
        predicted_aqi: result.predicted_aqi,
        category: result.category,
        model_used: result.model_used
      });
      console.log(`   📊 ${model}: ${result.predicted_aqi} (${result.category}) - Used: ${result.model_used}`);
    }
  }
  
  return results;
}

async function testCityVariations() {
  console.log('\n🏙️ Testing City Name Variations');
  
  const cityTests = [
    'Delhi', 'New Delhi', 'NCT', 'Delhi NCT',
    'Mumbai', 'Bombay', 
    'Bangalore', 'Bengaluru',
    'Chennai', 'Madras',
    'Kolkata', 'Calcutta'
  ];
  
  const results = [];
  
  for (const city of cityTests) {
    console.log(`\n📍 Testing: ${city}`);
    const result = await testAPI('/predict', {
      method: 'POST',
      body: JSON.stringify({
        city: city,
        date: '2024-01-15',
        model_type: 'auto'
      })
    });
    
    if (result) {
      results.push({
        original_city: city,
        normalized_city: result.city,
        predicted_aqi: result.predicted_aqi,
        category: result.category
      });
      console.log(`   📊 ${city} -> ${result.city}: ${result.predicted_aqi} (${result.category})`);
    }
  }
  
  return results;
}

async function testDataQuality() {
  console.log('\n📊 Testing Data Quality and Availability');
  
  const cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'];
  
  for (const city of cities) {
    console.log(`\n🔍 Checking data for: ${city}`);
    
    // Test debug endpoint
    const debugData = await testAPI(`/debug/${city}`);
    if (debugData) {
      console.log(`   📈 Records: ${debugData.total_records}`);
      console.log(`   📅 Date Range: ${debugData.date_range.start} to ${debugData.date_range.end}`);
      console.log(`   🏭 Stations: ${debugData.stations_available}`);
    }
    
    // Test station predictions
    const stationData = await testAPI('/predict/stations', {
      method: 'POST',
      body: JSON.stringify({
        city: city,
        date: '2024-01-15',
        model_type: 'auto',
        include_city_level: true
      })
    });
    
    if (stationData) {
      console.log(`   🏭 Station predictions: ${stationData.total_predictions}`);
      console.log(`   📍 Individual stations: ${stationData.station_predictions}`);
    }
  }
}

async function testFuturePredictions() {
  console.log('\n📅 Testing Future Predictions');
  
  const cities = ['Delhi', 'Mumbai', 'Bangalore'];
  
  for (const city of cities) {
    console.log(`\n🔮 Future predictions for: ${city}`);
    
    for (const days of [3, 7, 14]) {
      const result = await testAPI(`/predict/${city}/future?days=${days}`);
      if (result && result.predictions) {
        const aqis = result.predictions.map(p => p.predicted_aqi);
        const avg = aqis.reduce((a, b) => a + b, 0) / aqis.length;
        console.log(`   ${days} days: Avg AQI ${avg.toFixed(1)} (${aqis.length} predictions)`);
      }
    }
  }
}

async function testRealTimeComparison() {
  console.log('\n⚡ Testing Real-time Comparisons');
  
  const cities = ['Delhi', 'Mumbai', 'Bangalore'];
  
  for (const city of cities) {
    console.log(`\n🔍 Real-time comparison for: ${city}`);
    const result = await testAPI(`/realtime/${city}`);
    
    if (result) {
      console.log(`   📊 Current: ${result.realtime.aqi} (${result.realtime.category})`);
      console.log(`   🔮 Predicted: ${result.predictions.today?.predicted_aqi || 'N/A'} (${result.accuracy.today}% accuracy)`);
      console.log(`   📈 Trend: ${result.trend_analysis.short_term}`);
    }
  }
}

async function testErrorScenarios() {
  console.log('\n🚨 Testing Error Scenarios');
  
  const errorTests = [
    { 
      endpoint: '/predict', 
      method: 'POST', 
      body: { city: 'NonExistentCity', date: '2024-01-15', model_type: 'auto' },
      description: 'Non-existent city'
    },
    { 
      endpoint: '/predict', 
      method: 'POST', 
      body: { city: 'Delhi', date: '2020-01-01', model_type: 'auto' },
      description: 'Past date prediction'
    },
    { 
      endpoint: '/predict', 
      method: 'POST', 
      body: { city: 'Delhi', date: '2024-01-15', model_type: 'invalid_model' },
      description: 'Invalid model type'
    },
    { 
      endpoint: '/cities/NonExistentCity/stations', 
      method: 'GET',
      description: 'Stations for non-existent city'
    }
  ];
  
  for (const test of errorTests) {
    console.log(`\n❓ Testing: ${test.description}`);
    await testAPI(test.endpoint, {
      method: test.method,
      body: test.body ? JSON.stringify(test.body) : undefined
    });
  }
}

async function runComprehensiveTests() {
  console.log('🚀 Starting Comprehensive API Tests...\n');
  
  try {
    // Basic API Health
    console.log('📋 1. BASIC API HEALTH');
    await testAPI('/health');
    await testAPI('/status');
    
    // System Configuration
    console.log('\n⚙️ 2. SYSTEM CONFIGURATION');
    await testAPI('/cities');
    await testAPI('/models/available');
    await testAPI('/models/performance');
    
    // Data Quality Tests
    console.log('\n📊 3. DATA QUALITY TESTS');
    await testDataQuality();
    
    // City Name Normalization
    console.log('\n🏙️ 4. CITY NAME NORMALIZATION TESTS');
    await testCityVariations();
    
    // Model Comparison
    console.log('\n🤖 5. MODEL COMPARISON TESTS');
    await testModelPredictions('Delhi');
    await testModelPredictions('Mumbai');
    
    // Prediction Features
    console.log('\n🔮 6. PREDICTION FEATURE TESTS');
    await testFuturePredictions();
    await testRealTimeComparison();
    
    // Current Data
    console.log('\n🌐 7. CURRENT DATA TESTS');
    await testAPI('/current-aqi');
    await testAPI('/cities/comparison');
    
    // Error Handling
    console.log('\n🚨 8. ERROR HANDLING TESTS');
    await testErrorScenarios();
    
    console.log('\n🎉 ALL COMPREHENSIVE TESTS COMPLETED!');
    
  } catch (error) {
    console.error('\n💥 TEST SUITE FAILED:', error);
  }
}

// Run comprehensive tests
runComprehensiveTests();

// Also export for use in other files
module.exports = {
  testAPI,
  testModelPredictions,
  testCityVariations,
  testDataQuality,
  runComprehensiveTests
};