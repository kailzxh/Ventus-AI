import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Create DataFrame from the data
data = [
{"City": "Delhi", "Year": 2015, "PM2.5": 180.2, "AQI": 450.1, "Season": "Winter"},
{"City": "Mumbai", "Year": 2015, "PM2.5": 120.5, "AQI": 320.8, "Season": "Winter"},
{"City": "Kolkata", "Year": 2015, "PM2.5": 150.3, "AQI": 380.4, "Season": "Winter"},
{"City": "Chennai", "Year": 2015, "PM2.5": 95.7, "AQI": 280.2, "Season": "Winter"},
{"City": "Bangalore", "Year": 2015, "PM2.5": 75.4, "AQI": 245.6, "Season": "Winter"},
{"City": "Hyderabad", "Year": 2015, "PM2.5": 85.2, "AQI": 260.3, "Season": "Winter"},
{"City": "Delhi", "Year": 2016, "PM2.5": 175.8, "AQI": 445.2, "Season": "Winter"},
{"City": "Mumbai", "Year": 2016, "PM2.5": 118.3, "AQI": 315.4, "Season": "Winter"},
{"City": "Kolkata", "Year": 2016, "PM2.5": 148.7, "AQI": 375.8, "Season": "Winter"},
{"City": "Chennai", "Year": 2016, "PM2.5": 93.2, "AQI": 275.9, "Season": "Winter"},
{"City": "Bangalore", "Year": 2016, "PM2.5": 73.8, "AQI": 240.1, "Season": "Winter"},
{"City": "Hyderabad", "Year": 2016, "PM2.5": 83.4, "AQI": 255.7, "Season": "Winter"},
{"City": "Delhi", "Year": 2017, "PM2.5": 178.5, "AQI": 448.3, "Season": "Winter"},
{"City": "Mumbai", "Year": 2017, "PM2.5": 119.7, "AQI": 318.2, "Season": "Winter"},
{"City": "Kolkata", "Year": 2017, "PM2.5": 149.8, "AQI": 378.5, "Season": "Winter"},
{"City": "Chennai", "Year": 2017, "PM2.5": 94.1, "AQI": 277.4, "Season": "Winter"},
{"City": "Bangalore", "Year": 2017, "PM2.5": 74.6, "AQI": 242.8, "Season": "Winter"},
{"City": "Hyderabad", "Year": 2017, "PM2.5": 84.3, "AQI": 258.1, "Season": "Winter"},
{"City": "Delhi", "Year": 2018, "PM2.5": 182.1, "AQI": 452.6, "Season": "Winter"},
{"City": "Mumbai", "Year": 2018, "PM2.5": 121.4, "AQI": 322.1, "Season": "Winter"},
{"City": "Kolkata", "Year": 2018, "PM2.5": 151.2, "AQI": 381.7, "Season": "Winter"},
{"City": "Chennai", "Year": 2018, "PM2.5": 95.8, "AQI": 280.5, "Season": "Winter"},
{"City": "Bangalore", "Year": 2018, "PM2.5": 75.9, "AQI": 245.3, "Season": "Winter"},
{"City": "Hyderabad", "Year": 2018, "PM2.5": 85.7, "AQI": 261.4, "Season": "Winter"},
{"City": "Delhi", "Year": 2019, "PM2.5": 176.3, "AQI": 443.8, "Season": "Winter"},
{"City": "Mumbai", "Year": 2019, "PM2.5": 117.9, "AQI": 313.7, "Season": "Winter"},
{"City": "Kolkata", "Year": 2019, "PM2.5": 147.5, "AQI": 373.2, "Season": "Winter"},
{"City": "Chennai", "Year": 2019, "PM2.5": 92.4, "AQI": 273.8, "Season": "Winter"},
{"City": "Bangalore", "Year": 2019, "PM2.5": 72.1, "AQI": 237.9, "Season": "Winter"},
{"City": "Hyderabad", "Year": 2019, "PM2.5": 82.6, "AQI": 253.4, "Season": "Winter"},
{"City": "Delhi", "Year": 2020, "PM2.5": 174.7, "AQI": 441.2, "Season": "Winter"},
{"City": "Mumbai", "Year": 2020, "PM2.5": 116.8, "AQI": 311.5, "Season": "Winter"},
{"City": "Kolkata", "Year": 2020, "PM2.5": 146.9, "AQI": 371.8, "Season": "Winter"},
{"City": "Chennai", "Year": 2020, "PM2.5": 91.7, "AQI": 272.1, "Season": "Winter"},
{"City": "Bangalore", "Year": 2020, "PM2.5": 71.5, "AQI": 236.2, "Season": "Winter"},
{"City": "Hyderabad", "Year": 2020, "PM2.5": 81.9, "AQI": 251.8, "Season": "Winter"}
]

df = pd.DataFrame(data)

# Sort cities by average AQI to identify top polluted cities
avg_aqi_by_city = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
top_6_cities = avg_aqi_by_city.head(6).index.tolist()

# Filter data for top 6 cities
df_filtered = df[df['City'].isin(top_6_cities)]

# Create line chart showing AQI trends over time for top 6 polluted cities
fig = px.line(df_filtered, 
              x='Year', 
              y='AQI', 
              color='City',
              title='Top 6 Cities AQI Trends (2015-2020)',
              color_discrete_sequence=['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C'])

# Update layout with improved styling
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='AQI',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
)

# Update traces for better visualization and hover information
fig.update_traces(
    mode='lines+markers',
    marker=dict(size=6),
    line=dict(width=3),
    cliponaxis=False,
    hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>AQI: %{y:.1f}<extra></extra>'
)

# Update x-axis to show all years clearly
fig.update_xaxes(
    dtick=1,
    tickangle=0
)

# Update y-axis formatting
fig.update_yaxes(
    tickformat='.0f'
)

# Save the chart
fig.write_image('comprehensive_aqi_trends.png')