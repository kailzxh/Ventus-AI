// diagnose-models.js
const API_BASE = 'http://localhost:5000/api';

async function diagnoseModelIssues() {
  console.log('🔧 DIAGNOSING MODEL PREDICTION ISSUES\n');
  
  // Test the problematic cities with different models
  const testCases = [
    { city: 'Delhi', expected_range: [100, 400] },
    { city: 'Mumbai', expected_range: [80, 200] },
    { city: 'Bangalore', expected_range: [60, 150] },
    { city: 'Chennai', expected_range: [70, 180] },
    { city: 'Kolkata', expected_range: [100, 300] }
  ];
  
  const models = ['nf_vae', 'random_forest', 'gradient_boosting'];
  
  console.log('📋 MODEL PREDICTION ANALYSIS');
  console.log('=' .repeat(80));
  
  for (const testCase of testCases) {
    console.log(`\n🏙️ ${testCase.city} (Expected: ${testCase.expected_range[0]}-${testCase.expected_range[1]})`);
    console.log('-'.repeat(60));
    
    for (const model of models) {
      try {
        const response = await fetch(`${API_BASE}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            city: testCase.city,
            date: '2024-01-15',
            model_type: model
          })
        });
        
        const data = await response.json();
        const is_reasonable = data.predicted_aqi >= testCase.expected_range[0] && 
                             data.predicted_aqi <= testCase.expected_range[1];
        const status = is_reasonable ? '✅ REASONABLE' : '❌ SUSPICIOUS';
        
        console.log(`   ${model.padEnd(18)}: ${data.predicted_aqi.toFixed(1).padEnd(6)} (${data.category.padEnd(12)}) ${status}`);
        
      } catch (error) {
        console.log(`   ${model.padEnd(18)}: ❌ FAILED - ${error.message}`);
      }
    }
  }
  
  // Test data availability
  console.log('\n\n📊 DATA AVAILABILITY CHECK');
  console.log('=' .repeat(80));
  
  for (const testCase of testCases) {
    try {
      const response = await fetch(`${API_BASE}/debug/${testCase.city}`);
      const data = await response.json();
      
      console.log(`\n🏙️ ${testCase.city}:`);
      console.log(`   📈 Records: ${data.total_records}`);
      console.log(`   📅 Date Range: ${data.date_range.start} to ${data.date_range.end}`);
      console.log(`   🏭 Stations: ${data.stations_available}`);
      console.log(`   📋 Columns: ${data.available_columns?.length || 0} features`);
      
    } catch (error) {
      console.log(`\n🏙️ ${testCase.city}: ❌ Debug endpoint failed`);
    }
  }
}

diagnoseModelIssues();