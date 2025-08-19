// Application Data
const appData = {
  "cities": ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Ahmedabad", "Pune", "Lucknow", "Kanpur", "Patna", "Gurgaon", "Agra", "Jaipur", "Coimbatore", "Surat", "Indore", "Bhopal"],
  "current_aqi": {
    "Delhi": {"aqi": 387, "category": "Very Poor", "pm25": 185.2, "pm10": 298.5},
    "Mumbai": {"aqi": 152, "category": "Moderate", "pm25": 72.4, "pm10": 134.2},
    "Bangalore": {"aqi": 89, "category": "Satisfactory", "pm25": 42.1, "pm10": 78.3},
    "Chennai": {"aqi": 98, "category": "Satisfactory", "pm25": 46.8, "pm10": 87.1},
    "Kolkata": {"aqi": 234, "category": "Poor", "pm25": 112.3, "pm10": 187.9},
    "Hyderabad": {"aqi": 123, "category": "Moderate", "pm25": 58.7, "pm10": 102.4},
    "Ahmedabad": {"aqi": 167, "category": "Moderate", "pm25": 79.8, "pm10": 145.6},
    "Pune": {"aqi": 76, "category": "Satisfactory", "pm25": 36.2, "pm10": 68.9},
    "Lucknow": {"aqi": 298, "category": "Poor", "pm25": 142.6, "pm10": 234.1},
    "Kanpur": {"aqi": 365, "category": "Very Poor", "pm25": 174.8, "pm10": 287.3},
    "Patna": {"aqi": 287, "category": "Poor", "pm25": 137.2, "pm10": 223.8},
    "Gurgaon": {"aqi": 321, "category": "Very Poor", "pm25": 153.7, "pm10": 251.4},
    "Agra": {"aqi": 198, "category": "Moderate", "pm25": 94.6, "pm10": 164.2},
    "Jaipur": {"aqi": 145, "category": "Moderate", "pm25": 69.3, "pm10": 126.8},
    "Coimbatore": {"aqi": 67, "category": "Satisfactory", "pm25": 32.1, "pm10": 59.4},
    "Surat": {"aqi": 134, "category": "Moderate", "pm25": 64.0, "pm10": 118.7},
    "Indore": {"aqi": 112, "category": "Moderate", "pm25": 53.4, "pm10": 96.2},
    "Bhopal": {"aqi": 95, "category": "Satisfactory", "pm25": 45.3, "pm10": 83.7}
  },
  "historical_data": [
    {"date": "2024-01-01", "city": "Delhi", "aqi": 420, "pm25": 201.2, "season": "Winter"},
    {"date": "2024-01-01", "city": "Mumbai", "aqi": 165, "pm25": 78.9, "season": "Winter"},
    {"date": "2024-01-01", "city": "Bangalore", "aqi": 92, "pm25": 44.1, "season": "Winter"},
    {"date": "2024-02-01", "city": "Delhi", "aqi": 398, "pm25": 190.5, "season": "Winter"},
    {"date": "2024-02-01", "city": "Mumbai", "aqi": 158, "pm25": 75.6, "season": "Winter"},
    {"date": "2024-02-01", "city": "Bangalore", "aqi": 88, "pm25": 42.1, "season": "Winter"},
    {"date": "2024-03-01", "city": "Delhi", "aqi": 345, "pm25": 165.2, "season": "Spring"},
    {"date": "2024-03-01", "city": "Mumbai", "aqi": 142, "pm25": 67.8, "season": "Spring"},
    {"date": "2024-03-01", "city": "Bangalore", "aqi": 78, "pm25": 37.2, "season": "Spring"},
    {"date": "2024-04-01", "city": "Delhi", "aqi": 298, "pm25": 142.6, "season": "Spring"},
    {"date": "2024-04-01", "city": "Mumbai", "aqi": 128, "pm25": 61.1, "season": "Spring"},
    {"date": "2024-04-01", "city": "Bangalore", "aqi": 72, "pm25": 34.3, "season": "Spring"},
    {"date": "2024-05-01", "city": "Delhi", "aqi": 267, "pm25": 127.8, "season": "Spring"},
    {"date": "2024-05-01", "city": "Mumbai", "aqi": 115, "pm25": 54.9, "season": "Spring"},
    {"date": "2024-05-01", "city": "Bangalore", "aqi": 65, "pm25": 31.1, "season": "Spring"},
    {"date": "2024-06-01", "city": "Delhi", "aqi": 189, "pm25": 90.4, "season": "Summer"},
    {"date": "2024-06-01", "city": "Mumbai", "aqi": 98, "pm25": 46.8, "season": "Summer"},
    {"date": "2024-06-01", "city": "Bangalore", "aqi": 58, "pm25": 27.7, "season": "Summer"},
    {"date": "2024-07-01", "city": "Delhi", "aqi": 156, "pm25": 74.6, "season": "Summer"},
    {"date": "2024-07-01", "city": "Mumbai", "aqi": 89, "pm25": 42.5, "season": "Summer"},
    {"date": "2024-07-01", "city": "Bangalore", "aqi": 52, "pm25": 24.8, "season": "Summer"},
    {"date": "2024-08-01", "city": "Delhi", "aqi": 143, "pm25": 68.3, "season": "Summer"},
    {"date": "2024-08-01", "city": "Mumbai", "aqi": 82, "pm25": 39.2, "season": "Summer"},
    {"date": "2024-08-01", "city": "Bangalore", "aqi": 48, "pm25": 22.9, "season": "Summer"}
  ],
  "model_performance": {
    "Random_Forest": {"rmse": 28.5, "mae": 21.3, "r2": 0.892},
    "Gradient_Boosting": {"rmse": 31.2, "mae": 23.7, "r2": 0.871},
    "NF_VAE": {"rmse": 24.1, "mae": 18.9, "r2": 0.925},
    "Linear_Regression": {"rmse": 45.6, "mae": 34.2, "r2": 0.743}
  },
  "pollutant_correlations": {
    "PM2.5_AQI": 0.94,
    "PM10_AQI": 0.87,
    "NO2_AQI": 0.72,
    "SO2_AQI": 0.58,
    "CO_AQI": 0.65,
    "O3_AQI": 0.43
  },
  "dataset_stats": {
    "total_records": 39456,
    "cities_count": 18,
    "date_range": "2015-01-01 to 2020-12-31",
    "avg_aqi": 185.4,
    "max_aqi": 500.0
  }
};

// Chart colors
const chartColors = ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325', '#944454', '#13343B'];

// Global chart instances
let chartsInstances = {};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    updateCurrentDateTime();
    setInterval(updateCurrentDateTime, 60000); // Update every minute
    
    setupTabs();
    renderOverviewTab();
    
    // Initialize other tabs but don't render charts until they're active
    setTimeout(() => {
        renderDataExplorer();
        renderPredictionsTab();
        renderCityComparison();
        renderAnalyticsTab();
    }, 100);
}

function updateCurrentDateTime() {
    const now = new Date();
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'Asia/Kolkata'
    };
    document.getElementById('current-date').textContent = now.toLocaleDateString('en-IN', options) + ' IST';
}

function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const tabId = this.getAttribute('data-tab');
            
            // Remove active class from all buttons and panels
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanels.forEach(panel => panel.classList.remove('active'));
            
            // Add active class to clicked button and corresponding panel
            this.classList.add('active');
            const targetPanel = document.getElementById(tabId);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
            
            // Trigger chart rendering for specific tabs
            if (tabId === 'explorer') {
                setTimeout(renderDataExplorer, 100);
            } else if (tabId === 'comparison') {
                setTimeout(renderCityComparison, 100);
            } else if (tabId === 'analytics') {
                setTimeout(renderAnalyticsTab, 100);
            }
        });
    });
}

function renderOverviewTab() {
    renderAQIStatusGrid();
}

function renderAQIStatusGrid() {
    const grid = document.getElementById('aqi-status-grid');
    if (!grid) return;
    
    grid.innerHTML = '';
    
    Object.entries(appData.current_aqi).forEach(([city, data]) => {
        const card = createAQICard(city, data);
        grid.appendChild(card);
    });
}

function createAQICard(city, data) {
    const card = document.createElement('div');
    const categoryClass = getCategoryClass(data.category);
    
    card.className = `aqi-card ${categoryClass}`;
    card.innerHTML = `
        <div class="aqi-card-header">
            <div class="aqi-city-name">${city}</div>
            <div class="aqi-value">${data.aqi}</div>
        </div>
        <div class="aqi-category ${categoryClass}">${data.category}</div>
        <div class="aqi-details">
            <span>PM2.5: ${data.pm25}</span>
            <span>PM10: ${data.pm10}</span>
        </div>
    `;
    
    return card;
}

function getCategoryClass(category) {
    const categoryMap = {
        'Good': 'good',
        'Satisfactory': 'good',
        'Moderate': 'moderate',
        'Poor': 'poor',
        'Very Poor': 'very-poor',
        'Severe': 'severe'
    };
    return categoryMap[category] || 'moderate';
}

function renderDataExplorer() {
    renderTrendsChart();
    renderSeasonalChart();
    renderDistributionChart();
    renderCorrelations();
}

function renderTrendsChart() {
    const canvas = document.getElementById('trends-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Prepare data for top cities
    const topCities = ['Delhi', 'Mumbai', 'Bangalore'];
    const datasets = topCities.map((city, index) => {
        const cityData = appData.historical_data.filter(d => d.city === city);
        return {
            label: city,
            data: cityData.map(d => d.aqi),
            borderColor: chartColors[index],
            backgroundColor: chartColors[index] + '20',
            tension: 0.4,
            fill: false
        };
    });
    
    if (chartsInstances['trends']) {
        chartsInstances['trends'].destroy();
    }
    
    chartsInstances['trends'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: appData.historical_data.filter(d => d.city === 'Delhi').map(d => {
                const date = new Date(d.date);
                return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
            }),
            datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'AQI Trends Over Time'
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'AQI Value'
                    }
                }
            }
        }
    });
}

function renderSeasonalChart() {
    const canvas = document.getElementById('seasonal-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Calculate seasonal averages
    const seasonalData = {};
    appData.historical_data.forEach(record => {
        if (!seasonalData[record.season]) {
            seasonalData[record.season] = [];
        }
        seasonalData[record.season].push(record.aqi);
    });
    
    const seasons = Object.keys(seasonalData);
    const averages = seasons.map(season => 
        Math.round(seasonalData[season].reduce((a, b) => a + b, 0) / seasonalData[season].length)
    );
    
    if (chartsInstances['seasonal']) {
        chartsInstances['seasonal'].destroy();
    }
    
    chartsInstances['seasonal'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: seasons,
            datasets: [{
                label: 'Average AQI',
                data: averages,
                backgroundColor: chartColors.slice(0, seasons.length)
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Average AQI'
                    }
                }
            }
        }
    });
}

function renderDistributionChart() {
    const canvas = document.getElementById('distribution-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Calculate AQI distribution by category
    const categories = {
        'Good (0-50)': 0,
        'Satisfactory (51-100)': 0,
        'Moderate (101-200)': 0,
        'Poor (201-300)': 0,
        'Very Poor (301-400)': 0,
        'Severe (401-500)': 0
    };
    
    Object.values(appData.current_aqi).forEach(data => {
        const aqi = data.aqi;
        if (aqi <= 50) categories['Good (0-50)']++;
        else if (aqi <= 100) categories['Satisfactory (51-100)']++;
        else if (aqi <= 200) categories['Moderate (101-200)']++;
        else if (aqi <= 300) categories['Poor (201-300)']++;
        else if (aqi <= 400) categories['Very Poor (301-400)']++;
        else categories['Severe (401-500)']++;
    });
    
    if (chartsInstances['distribution']) {
        chartsInstances['distribution'].destroy();
    }
    
    chartsInstances['distribution'] = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: Object.keys(categories),
            datasets: [{
                data: Object.values(categories),
                backgroundColor: chartColors
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function renderCorrelations() {
    const grid = document.getElementById('correlations-grid');
    if (!grid) return;
    
    grid.innerHTML = '';
    
    Object.entries(appData.pollutant_correlations).forEach(([pollutant, correlation]) => {
        const item = document.createElement('div');
        item.className = 'correlation-item';
        item.innerHTML = `
            <div class="correlation-value">${correlation.toFixed(2)}</div>
            <div class="correlation-label">${pollutant.replace('_', ' vs ')}</div>
        `;
        grid.appendChild(item);
    });
}

function renderPredictionsTab() {
    renderModelMetrics();
}

function renderModelMetrics() {
    const container = document.getElementById('model-metrics');
    if (!container) return;
    
    container.innerHTML = '';
    
    Object.entries(appData.model_performance).forEach(([model, metrics]) => {
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `
            <div class="metric-name">${model.replace('_', ' ')}</div>
            <div class="metric-value">R² ${metrics.r2.toFixed(3)}</div>
            <div class="metric-details">
                RMSE: ${metrics.rmse}<br>
                MAE: ${metrics.mae}
            </div>
        `;
        container.appendChild(card);
    });
}

function generatePrediction() {
    const city = document.getElementById('predict-city').value;
    const date = document.getElementById('predict-date').value;
    const season = document.getElementById('predict-season').value;
    
    // Simulate NF-VAE prediction logic
    const baseAQI = appData.current_aqi[city]?.aqi || 150;
    const seasonMultiplier = {
        'Winter': 1.4,
        'Spring': 1.1,
        'Summer': 0.7,
        'Monsoon': 0.8
    };
    
    const predictedAQI = Math.round(baseAQI * (seasonMultiplier[season] || 1.0) * (0.9 + Math.random() * 0.2));
    const confidence = [Math.round(predictedAQI * 0.85), Math.round(predictedAQI * 1.15)];
    
    const category = getAQICategory(predictedAQI);
    
    // Update prediction results
    document.getElementById('predicted-aqi').textContent = predictedAQI;
    document.getElementById('predicted-category').textContent = category;
    document.getElementById('predicted-category').className = `prediction-category ${getCategoryClass(category)}`;
    document.getElementById('confidence-range').textContent = `${confidence[0]} - ${confidence[1]}`;
    
    document.getElementById('prediction-results').classList.remove('hidden');
    
    // Render prediction chart
    renderPredictionChart(city, predictedAQI);
}

function getAQICategory(aqi) {
    if (aqi <= 50) return 'Good';
    if (aqi <= 100) return 'Satisfactory';
    if (aqi <= 200) return 'Moderate';
    if (aqi <= 300) return 'Poor';
    if (aqi <= 400) return 'Very Poor';
    return 'Severe';
}

function renderPredictionChart(city, predictedAQI) {
    const canvas = document.getElementById('prediction-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Get historical data for the city
    const historicalData = appData.historical_data.filter(d => d.city === city);
    const dates = historicalData.map(d => {
        const date = new Date(d.date);
        return date.toLocaleDateString('en-US', { month: 'short' });
    });
    const values = historicalData.map(d => d.aqi);
    
    // Add prediction
    dates.push('Dec (Predicted)');
    
    if (chartsInstances['prediction']) {
        chartsInstances['prediction'].destroy();
    }
    
    const historicalDataset = Array(values.length).fill(null);
    historicalDataset.splice(0, values.length, ...values);
    
    const predictionDataset = Array(dates.length).fill(null);
    predictionDataset[predictionDataset.length - 1] = predictedAQI;
    
    chartsInstances['prediction'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Historical AQI',
                data: historicalDataset,
                borderColor: chartColors[0],
                backgroundColor: chartColors[0] + '20',
                tension: 0.4
            }, {
                label: 'Predicted AQI',
                data: predictionDataset,
                borderColor: chartColors[1],
                backgroundColor: chartColors[1] + '20',
                pointBackgroundColor: chartColors[1],
                pointRadius: 8,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `${city} AQI Prediction`
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'AQI Value'
                    }
                }
            }
        }
    });
}

function renderCityComparison() {
    renderComparisonChart();
    renderRankingsTable();
}

function renderComparisonChart() {
    const canvas = document.getElementById('comparison-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    const cities = Object.keys(appData.current_aqi);
    const aqiValues = cities.map(city => appData.current_aqi[city].aqi);
    
    if (chartsInstances['comparison']) {
        chartsInstances['comparison'].destroy();
    }
    
    chartsInstances['comparison'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: cities,
            datasets: [{
                label: 'Current AQI',
                data: aqiValues,
                backgroundColor: chartColors[0],
                borderColor: chartColors[0],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'AQI Value'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45
                    }
                }
            }
        }
    });
}

function renderRankingsTable() {
    const container = document.getElementById('rankings-table');
    if (!container) return;
    
    // Sort cities by AQI (worst first)
    const sortedCities = Object.entries(appData.current_aqi)
        .sort(([,a], [,b]) => b.aqi - a.aqi);
    
    const table = document.createElement('table');
    table.innerHTML = `
        <thead>
            <tr>
                <th>Rank</th>
                <th>City</th>
                <th>AQI</th>
                <th>Category</th>
                <th>PM2.5</th>
                <th>PM10</th>
            </tr>
        </thead>
        <tbody>
            ${sortedCities.map(([city, data], index) => `
                <tr>
                    <td class="rank-number">#${index + 1}</td>
                    <td>${city}</td>
                    <td>${data.aqi}</td>
                    <td><span class="aqi-category ${getCategoryClass(data.category)}">${data.category}</span></td>
                    <td>${data.pm25}</td>
                    <td>${data.pm10}</td>
                </tr>
            `).join('')}
        </tbody>
    `;
    
    container.innerHTML = '';
    container.appendChild(table);
}

function renderAnalyticsTab() {
    renderAnalyticsChart();
    renderFeatureImportance();
    renderAnomalies();
}

function renderAnalyticsChart() {
    const canvas = document.getElementById('analytics-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Trend analysis - moving average
    const delhiData = appData.historical_data.filter(d => d.city === 'Delhi');
    const movingAvg = calculateMovingAverage(delhiData.map(d => d.aqi), 3);
    
    if (chartsInstances['analytics']) {
        chartsInstances['analytics'].destroy();
    }
    
    chartsInstances['analytics'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: delhiData.map(d => {
                const date = new Date(d.date);
                return date.toLocaleDateString('en-US', { month: 'short' });
            }),
            datasets: [{
                label: 'Actual AQI',
                data: delhiData.map(d => d.aqi),
                borderColor: chartColors[0],
                backgroundColor: chartColors[0] + '20',
                tension: 0.4
            }, {
                label: 'Moving Average (3-period)',
                data: movingAvg,
                borderColor: chartColors[1],
                backgroundColor: chartColors[1] + '20',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Delhi AQI Trend Analysis'
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'AQI Value'
                    }
                }
            }
        }
    });
}

function calculateMovingAverage(data, period) {
    const result = [];
    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
            result.push(null);
        } else {
            const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
            result.push(Math.round(sum / period));
        }
    }
    return result;
}

function renderFeatureImportance() {
    const container = document.getElementById('feature-importance');
    if (!container) return;
    
    const features = [
        { name: 'PM2.5', score: 0.94 },
        { name: 'PM10', score: 0.87 },
        { name: 'NO2', score: 0.72 },
        { name: 'CO', score: 0.65 },
        { name: 'SO2', score: 0.58 },
        { name: 'O3', score: 0.43 }
    ];
    
    container.innerHTML = features.map(feature => `
        <div class="feature-item">
            <span class="feature-name">${feature.name}</span>
            <span class="feature-score">${feature.score.toFixed(2)}</span>
        </div>
    `).join('');
}

function renderAnomalies() {
    const container = document.getElementById('anomaly-list');
    if (!container) return;
    
    const anomalies = [
        { date: '2024-01-15', description: 'Delhi AQI spike to 450 (expected: 380)' },
        { date: '2024-03-22', description: 'Mumbai unusual PM2.5 levels during monsoon' },
        { date: '2024-06-08', description: 'Bangalore AQI drop below predicted summer baseline' },
        { date: '2024-07-12', description: 'Kolkata ozone levels 40% above seasonal average' }
    ];
    
    container.innerHTML = anomalies.map(anomaly => `
        <div class="anomaly-item">
            <div class="anomaly-date">${anomaly.date}</div>
            <div class="anomaly-description">${anomaly.description}</div>
        </div>
    `).join('');
}

// Export functionality
function exportData() {
    const dataStr = JSON.stringify(appData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'aqi_data.json';
    link.click();
}

// Make functions globally available
window.generatePrediction = generatePrediction;
window.exportData = exportData;