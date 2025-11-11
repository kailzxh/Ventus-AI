# backend/data_loader.py
import pandas as pd
import numpy as np
import kaggle
import os
import requests
import json
import glob
from datetime import datetime, timedelta
from config import Config, INDIAN_CITIES

class ComprehensiveAQIDataLoader:
    def __init__(self):
        self.datasets = {}
        self.realtime_data = {}
        self.stations_metadata = None  # Add this for station metadata storage

    def _debug_station_mapping(self):
        """Debug method to show station mapping status"""
        if self.stations_metadata is None:
            print("❌ No station metadata available")
            return
        
        print("\n🔍 STATION MAPPING DEBUG INFO:")
        print(f"📊 Total stations in metadata: {len(self.stations_metadata)}")
        
        if 'StationId' in self.stations_metadata.columns:
            sample_ids = self.stations_metadata['StationId'].dropna().head(10).tolist()
            print(f"📍 Sample StationIds: {sample_ids}")
        
        if 'StationName' in self.stations_metadata.columns:
            sample_names = self.stations_metadata['StationName'].dropna().head(10).tolist()
            print(f"🏷️  Sample StationNames: {sample_names}")
        
        if 'City' in self.stations_metadata.columns:
            cities = self.stations_metadata['City'].dropna().unique()
            print(f"🏙️  Cities in metadata: {len(cities)} - {list(cities)[:10]}...")



    def load_historical_data(self):
        """Load ALL historical AQI data from multiple Kaggle datasets for 2015-2024"""
        print("📥 Loading Comprehensive Historical AQI Data (2015-2024)...")
        
        # Set Kaggle credentials
        os.environ['KAGGLE_USERNAME'] = 'kailzxh'
        os.environ['KAGGLE_KEY'] = 'a0601d99ed2c238727c9cc1ba4bb562c'
        
        # Define datasets with their actual file structures
        datasets_to_load = [
            {
                'name': 'Indian AQI 2015-2020',
                'id': 'rohanrao/air-quality-data-in-india',
                'download_name': 'india-aqi-2015-2020',
                'expected_files': ['city_day.csv', 'city_hour.csv', 'station_day.csv', 'station_hour.csv', 'stations.csv'],
                'date_range': '2015-2020'
            },
            {
                'name': 'Extended Indian AQI 2015-2024',
                'id': 'ankushpanday1/air-quality-data-in-india-2015-2024',
                'download_name': 'india-aqi-2015-2024', 
                'expected_files': ['city_day.csv', 'city_hour.csv', 'station_day.csv', 'station_hour.csv', 'stations.csv', 'AQI_Data.csv', 'data.csv'],
                'date_range': '2015-2024'
            }
        ]
        
        all_loaded_data = []
        dataset_files_summary = {}
        
        for dataset_info in datasets_to_load:
            try:
                print(f"\n🔍 Downloading: {dataset_info['name']}")
                print(f"   Dataset ID: {dataset_info['id']}")
                
                # Download dataset
                download_path = f"./data/raw/{dataset_info['download_name']}"
                os.makedirs(download_path, exist_ok=True)
                
                kaggle.api.dataset_download_files(
                    dataset_info['id'],
                    path=download_path,
                    unzip=True,
                    quiet=False
                )
                
                # Load all files from this dataset
                dataset_files = self._load_all_files_from_dataset(download_path, dataset_info)
                dataset_files_summary[dataset_info['name']] = dataset_files
                
                if dataset_files:
                    all_loaded_data.extend(dataset_files)
                    print(f"   ✅ Successfully loaded {len(dataset_files)} files")
                else:
                    print(f"   ⚠️  No files loaded from {dataset_info['name']}")
                        
            except Exception as e:
                print(f"   ❌ Failed to load {dataset_info['name']}: {e}")
                continue
        
        # Print comprehensive summary
        self._print_dataset_summary(dataset_files_summary)
        
        if all_loaded_data:
            # Merge all data with intelligent handling
            combined_df = self._intelligently_merge_datasets(all_loaded_data)
            print(f"\n✅ Final Combined Dataset: {len(combined_df):,} records")
        else:
            print("🔄 No datasets loaded, generating fallback data...")
            combined_df = self._generate_comprehensive_fallback_data()
        
        # Apply filters and enhancements
        combined_df = self._apply_data_enhancements(combined_df)
        
        # Store station metadata separately for easy access
        self._extract_station_metadata(all_loaded_data)
        
        # Print final statistics
        self._print_final_statistics(combined_df)
        self._extract_station_metadata(all_loaded_data)
    
    # Debug station mapping
        self._debug_station_mapping()
    
    # Print final statistics
        self._print_final_statistics(combined_df)
        return combined_df

    def _extract_station_metadata(self, all_loaded_data):
        """Extract and store station metadata separately with better merging"""
        print("\n📍 Extracting station metadata...")
        
        station_metadata_dfs = []
        
        for file_info in all_loaded_data:
            if file_info['file_type'] == 'station_metadata':
                df = file_info['data']
                # Ensure we have some meaningful data
                if len(df) > 0 and ('StationId' in df.columns or 'StationName' in df.columns):
                    station_metadata_dfs.append(df)
                    print(f"   ✅ Found {len(df)} station records from {file_info['filename']}")
        
        if station_metadata_dfs:
            # Merge all station metadata
            self.stations_metadata = pd.concat(station_metadata_dfs, ignore_index=True, sort=False)
            
            # Remove duplicates based on StationId or StationName
            if 'StationId' in self.stations_metadata.columns:
                self.stations_metadata = self.stations_metadata.drop_duplicates(subset=['StationId'], keep='first')
            elif 'StationName' in self.stations_metadata.columns:
                self.stations_metadata = self.stations_metadata.drop_duplicates(subset=['StationName'], keep='first')
            else:
                self.stations_metadata = self.stations_metadata.drop_duplicates(subset=['Station'], keep='first')
            
            # Fill missing critical columns
            if 'StationId' not in self.stations_metadata.columns and 'Station' in self.stations_metadata.columns:
                self.stations_metadata['StationId'] = self.stations_metadata['Station']
            
            if 'StationName' not in self.stations_metadata.columns and 'Station' in self.stations_metadata.columns:
                self.stations_metadata['StationName'] = self.stations_metadata['Station']
            
            print(f"✅ Loaded station metadata for {len(self.stations_metadata)} unique stations")
            
            # Print sample of station names for verification
            if 'StationName' in self.stations_metadata.columns:
                sample_stations = self.stations_metadata['StationName'].dropna().head(10).tolist()
                print(f"   📍 Sample stations: {sample_stations}")
                
        else:
            print("⚠️  No station metadata found, creating from station data...")
            self.stations_metadata = self._create_station_metadata_from_data(all_loaded_data)

    def _create_station_metadata_from_data(self, all_loaded_data):
        """Create station metadata from available station data with better extraction"""
        print("   Creating station metadata from available data...")
        
        station_data = []
        
        for file_info in all_loaded_data:
            if file_info['file_type'] in ['station_daily', 'station_hourly']:
                df = file_info['data']
                
                # Extract unique station information with more comprehensive approach
                station_cols = []
                if 'StationId' in df.columns:
                    station_cols.append('StationId')
                if 'Station' in df.columns:
                    station_cols.append('Station')
                if 'StationName' in df.columns:
                    station_cols.append('StationName')
                if 'City' in df.columns:
                    station_cols.append('City')
                if 'State' in df.columns:
                    station_cols.append('State')
                if 'Latitude' in df.columns:
                    station_cols.append('Latitude')
                if 'Longitude' in df.columns:
                    station_cols.append('Longitude')
                
                if station_cols:
                    # Get unique station combinations
                    station_info = df[station_cols].drop_duplicates()
                    
                    # Calculate some statistics for each station
                    if 'StationId' in df.columns:
                        station_stats = df.groupby('StationId').agg({
                            'AQI': ['count', 'mean', 'std'] if 'AQI' in df.columns else 'count',
                            'PM2.5': 'count' if 'PM2.5' in df.columns else None,
                            'PM10': 'count' if 'PM10' in df.columns else None
                        }).reset_index()
                        
                        # Flatten column names
                        station_stats.columns = ['_'.join(col).strip('_') for col in station_stats.columns.values]
                        station_stats = station_stats.rename(columns={'StationId_': 'StationId'})
                        
                        # Merge statistics with station info
                        if 'StationId' in station_info.columns:
                            station_info = pd.merge(station_info, station_stats, on='StationId', how='left')
                    
                    station_data.append(station_info)
                    print(f"      📊 Extracted {len(station_info)} stations from {file_info['filename']}")
        
        if station_data:
            combined_stations = pd.concat(station_data, ignore_index=True)
            
            # Remove duplicates - prefer records with more data
            if 'AQI_count' in combined_stations.columns:
                combined_stations = combined_stations.sort_values('AQI_count', ascending=False)
            
            if 'StationId' in combined_stations.columns:
                combined_stations = combined_stations.drop_duplicates(subset=['StationId'], keep='first')
            elif 'Station' in combined_stations.columns:
                combined_stations = combined_stations.drop_duplicates(subset=['Station'], keep='first')
            elif 'StationName' in combined_stations.columns:
                combined_stations = combined_stations.drop_duplicates(subset=['StationName'], keep='first')
            
            # Add missing columns with default values
            if 'StationName' not in combined_stations.columns:
                if 'Station' in combined_stations.columns:
                    combined_stations['StationName'] = combined_stations['Station']
                elif 'StationId' in combined_stations.columns:
                    combined_stations['StationName'] = combined_stations['StationId']
            
            if 'City' not in combined_stations.columns:
                combined_stations['City'] = 'Unknown'
            if 'State' not in combined_stations.columns:
                combined_stations['State'] = 'Unknown'
            if 'Latitude' not in combined_stations.columns:
                combined_stations['Latitude'] = np.nan
            if 'Longitude' not in combined_stations.columns:
                combined_stations['Longitude'] = np.nan
            
            print(f"   ✅ Created enhanced metadata for {len(combined_stations)} stations from data")
            return combined_stations
        else:
            print("   ⚠️  Could not create station metadata from data")
            return pd.DataFrame()

    def _load_all_files_from_dataset(self, download_path, dataset_info):
        """Load all files from a dataset directory with specific handling for each file type"""
        print(f"   📂 Scanning directory: {download_path}")
        
        loaded_files = []
        
        # Check if directory exists
        if not os.path.exists(download_path):
            print(f"   ❌ Directory not found: {download_path}")
            return loaded_files
            
        all_files = glob.glob(os.path.join(download_path, "*"))
        
        print(f"   Found {len(all_files)} files/directories:")
        for file_path in all_files:
            if os.path.isfile(file_path):
                print(f"     📄 {os.path.basename(file_path)}")
            else:
                print(f"     📁 {os.path.basename(file_path)}/")
        
        # Look for CSV files recursively
        csv_files = glob.glob(os.path.join(download_path, "**/*.csv"), recursive=True)
        print(f"   📊 Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            print(f"   🔍 Processing: {filename}")
            
            df = self._load_csv_file(csv_file)
            if df is not None and len(df) > 0:
                # Apply file-specific processing
                df = self._process_file_by_type(df, filename, dataset_info['name'])
                loaded_files.append({
                    'data': df,
                    'filename': filename,
                    'dataset': dataset_info['name'],
                    'file_type': self._classify_file_type(filename),
                    'file_path': csv_file
                })
                print(f"      ✅ Loaded {filename}: {len(df):,} records, {len(df.columns)} columns")
            else:
                print(f"      ❌ Failed to load {filename}")
        
        return loaded_files

    def _classify_file_type(self, filename):
        """Classify files by their type for specialized processing"""
        filename_lower = filename.lower()
        
        if 'city_day' in filename_lower:
            return 'city_daily'
        elif 'city_hour' in filename_lower:
            return 'city_hourly' 
        elif 'station_day' in filename_lower:
            return 'station_daily'
        elif 'station_hour' in filename_lower:
            return 'station_hourly'
        elif 'stations' in filename_lower:
            return 'station_metadata'
        elif 'aqi_data' in filename_lower:
            return 'aqi_data'
        elif 'data' in filename_lower:
            return 'general_data'
        else:
            return 'unknown'

    def _process_file_by_type(self, df, filename, dataset_name):
        """Apply specialized processing based on file type with better station handling"""
        file_type = self._classify_file_type(filename)
        
        # Add source information
        df['Source_Dataset'] = dataset_name
        df['Source_File'] = filename
        df['File_Type'] = file_type
        
        # Apply file-type specific processing
        if file_type in ['city_daily', 'city_hourly', 'station_daily', 'station_hourly']:
            df = self._standardize_pollutant_data(df, file_type)
            
            # For station data, try to enhance with station names from metadata
            if file_type in ['station_daily', 'station_hourly']:
                df = self._enhance_station_data_with_metadata(df)
                
        elif file_type == 'station_metadata':
            df = self._standardize_station_metadata(df)
        elif file_type == 'aqi_data':
            df = self._standardize_aqi_data(df)
        
        # Universal standardization
        df = self._apply_universal_standardization(df)
        
        return df

    def _enhance_station_data_with_metadata(self, df):
        """Enhance station data with proper station names from metadata"""
        if self.stations_metadata is None or self.stations_metadata.empty:
            return df
        
        print("      🔍 Enhancing station data with metadata...")
        
        # Create station ID to name mapping from metadata
        station_mapping = {}
        for _, station_row in self.stations_metadata.iterrows():
            station_id = station_row.get('StationId')
            station_name = station_row.get('StationName', station_row.get('Station'))
            if station_id and station_name:
                station_mapping[str(station_id)] = station_name
                # Also add reverse mapping for flexibility
                station_mapping[str(station_name)] = station_id
        
        # Try to enhance StationName column
        if 'StationId' in df.columns and 'StationName' not in df.columns:
            df['StationName'] = df['StationId'].map(station_mapping)
            enhanced_count = df['StationName'].notna().sum()
            if enhanced_count > 0:
                print(f"      ✅ Enhanced {enhanced_count} records with station names from metadata")
            else:
                print(f"      ⚠️  Could not enhance station names - no matches found")
        
        # Also try to enhance Station column if it contains IDs
        if 'Station' in df.columns:
            # Check if Station column contains IDs that can be mapped
            station_ids_in_data = df['Station'].dropna().unique()
            mapped_stations = 0
            
            for station_id in station_ids_in_data:
                if str(station_id) in station_mapping:
                    mapped_name = station_mapping[str(station_id)]
                    # Create a new StationName column or update existing
                    if 'StationName' not in df.columns:
                        df['StationName'] = df['Station']
                    df.loc[df['Station'] == station_id, 'StationName'] = mapped_name
                    mapped_stations += 1
            
            if mapped_stations > 0:
                print(f"      ✅ Mapped {mapped_stations} station IDs to names")
        
        return df

    def _standardize_pollutant_data(self, df, file_type):
        """Standardize pollutant data files with enhanced station handling and name mapping"""
        print(f"      Standardizing {file_type} data...")
        
        # Enhanced column mapping for better station identification
        column_mapping = {
            # Station Identifiers (Priority for station-level predictions)
            'station_id': 'StationId', 'StationId': 'StationId', 'STATIONID': 'StationId',
            'station': 'Station', 'Station': 'Station', 'STATION': 'Station', 
            'site': 'Station', 'Site': 'Station', 'SITE': 'Station',
            'location': 'Station', 'Location': 'Station', 'LOCATION': 'Station',
            'station_name': 'StationName', 'StationName': 'StationName', 'stationname': 'StationName',
            
            # City Identifiers
            'city': 'City', 'City': 'City', 'CITY': 'City', 'City Name': 'City',
            
            # Date/Time
            'date': 'Date', 'Date': 'Date', 'DATE': 'Date', 
            'datetime': 'Datetime', 'Datetime': 'Datetime', 'DateTime': 'Datetime',
            'time': 'Time', 'Time': 'Time', 'TIME': 'Time',
            
            # Primary pollutants
            'pm25': 'PM2.5', 'PM2.5': 'PM2.5', 'PM25': 'PM2.5', 'pm2.5': 'PM2.5',
            'pm10': 'PM10', 'PM10': 'PM10', 'pm': 'PM10',
            'no2': 'NO2', 'NO2': 'NO2', 'nitrogen dioxide': 'NO2',
            'so2': 'SO2', 'SO2': 'SO2', 'sulfur dioxide': 'SO2',
            'co': 'CO', 'CO': 'CO', 'carbon monoxide': 'CO',
            'o3': 'O3', 'O3': 'O3', 'ozone': 'O3',
            
            # AQI and categories
            'aqi': 'AQI', 'AQI': 'AQI', 'Air Quality Index': 'AQI', 'air_quality_index': 'AQI',
            'aqi_bucket': 'AQI_Bucket', 'AQI_Bucket': 'AQI_Bucket',
            
            # Meteorological
            'temperature': 'Temperature', 'Temperature': 'Temperature', 'temp': 'Temperature',
            'humidity': 'Humidity', 'Humidity': 'Humidity', 'hum': 'Humidity',
            'pressure': 'Pressure', 'Pressure': 'Pressure', 'pres': 'Pressure',
            'wind_speed': 'Wind_Speed', 'Wind_Speed': 'Wind_Speed', 'ws': 'Wind_Speed',
            
            # Geographical
            'state': 'State', 'State': 'State', 'STATE': 'State',
            'latitude': 'Latitude', 'Latitude': 'Latitude', 'lat': 'Latitude',
            'longitude': 'Longitude', 'Longitude': 'Longitude', 'lon': 'Longitude',
        }
        
        # Apply column mapping
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Ensure we have proper station identifiers
        if 'StationId' not in df.columns and 'Station' in df.columns:
            # Check if Station column contains IDs that should be mapped
            station_values = df['Station'].dropna().unique()
            has_station_ids = any(len(str(val)) <= 10 and any(c.isdigit() for c in str(val)) for val in station_values)
            
            if has_station_ids:
                df['StationId'] = df['Station']
                print(f"      🔄 Using Station column as StationId (contains IDs)")
            else:
                df['StationName'] = df['Station']
                print(f"      🔄 Using Station column as StationName (contains names)")
        
        # Ensure we have a primary station identifier for prediction system
        if 'Station' not in df.columns:
            if 'StationId' in df.columns:
                df['Station'] = df['StationId']
            elif 'StationName' in df.columns:
                df['Station'] = df['StationName']
            elif 'City' in df.columns:
                # For city-level data without stations, create synthetic station names
                df['Station'] = df['City'] + '_Central'
                print(f"      🔧 Created synthetic station names for city-level data")
        
        return df

    def _standardize_station_metadata(self, df):
        """Standardize station metadata files with comprehensive mapping"""
        print("      Standardizing station metadata...")
        
        column_mapping = {
            'station_id': 'StationId', 'StationId': 'StationId', 'STATIONID': 'StationId',
            'station': 'Station', 'Station': 'Station', 'STATION': 'Station',
            'name': 'StationName', 'station_name': 'StationName', 'Site': 'StationName',
            'city': 'City', 'City': 'City', 'CITY': 'City',
            'state': 'State', 'State': 'State', 'STATE': 'State',
            'latitude': 'Latitude', 'Latitude': 'Latitude', 'lat': 'Latitude',
            'longitude': 'Longitude', 'Longitude': 'Longitude', 'lon': 'Longitude',
            'altitude': 'Altitude', 'Altitude': 'Altitude', 'elevation': 'Altitude',
            'type': 'Station_Type', 'station_type': 'Station_Type', 'category': 'Station_Type',
            'status': 'Status', 'Status': 'Status', 'STATUS': 'Status'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Ensure critical columns exist with proper values
        if 'StationId' not in df.columns and 'Station' in df.columns:
            df['StationId'] = df['Station']
        
        if 'StationName' not in df.columns:
            if 'Station' in df.columns:
                df['StationName'] = df['Station']
            elif 'StationId' in df.columns:
                df['StationName'] = df['StationId']
        
        # Clean up station names - remove extra spaces and standardize
        if 'StationName' in df.columns:
            df['StationName'] = df['StationName'].astype(str).str.strip()
            # Remove any "Unknown" or placeholder names
            df['StationName'] = df['StationName'].replace(['Unknown', 'unknown', 'UNKNOWN', ''], np.nan)
        
        # Ensure City column has proper values
        if 'City' in df.columns:
            df['City'] = df['City'].astype(str).str.strip()
            df['City'] = df['City'].replace(['Unknown', 'unknown', 'UNKNOWN', ''], np.nan)
        
        print(f"      ✅ Standardized {len(df)} station metadata records")
        return df

    def _standardize_aqi_data(self, df):
        """Standardize AQI-specific data files"""
        print("      Standardizing AQI data...")
        return self._standardize_pollutant_data(df, 'aqi_data')

    def _apply_universal_standardization(self, df):
        """Apply universal standardization to all dataframes"""
        # Handle date/time columns
        for date_col in ['Date', 'Datetime']:
            if date_col in df.columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                except Exception as e:
                    print(f"      ⚠️  {date_col} conversion error: {e}")
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI', 'Temperature', 'Humidity', 'Pressure',
                          'Wind_Speed', 'Latitude', 'Longitude', 'Altitude']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def _load_csv_file(self, file_path):
        """Load CSV file with multiple encoding attempts"""
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                if len(df) > 0:
                    return df
            except Exception as e:
                continue
        
        # Final attempt with error handling
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', low_memory=False)
            return df if len(df) > 0 else None
        except Exception as e:
            print(f"      ❌ Failed to load {file_path}: {e}")
            return None

    def _intelligently_merge_datasets(self, all_loaded_data):
        """Intelligently merge all datasets preserving all features with station focus"""
        print("\n🔄 Intelligently merging datasets...")
        
        if not all_loaded_data:
            print("⚠️  No datasets to merge")
            return pd.DataFrame()
        
        # Prioritize station-level data
        station_data = []
        city_data = []
        other_data = []
        
        for file_info in all_loaded_data:
            if file_info['file_type'] in ['station_daily', 'station_hourly']:
                station_data.append(file_info)
            elif file_info['file_type'] in ['city_daily', 'city_hourly']:
                city_data.append(file_info)
            else:
                other_data.append(file_info)
        
        print(f"   📊 Found: {len(station_data)} station files, {len(city_data)} city files, {len(other_data)} other files")
        
        # Merge station data first (highest priority)
        merged_df = self._merge_station_data(station_data)
        
        # If no station data, use city data
        if merged_df.empty and city_data:
            print("   ⚠️  No station data found, using city-level data")
            merged_df = self._merge_city_data(city_data)
        
        # Add other data if available
        if not merged_df.empty and other_data:
            merged_df = self._merge_with_other_data(merged_df, other_data)
        
        if merged_df.empty:
            print("   ⚠️  All merge attempts failed, generating fallback data")
            merged_df = self._generate_comprehensive_fallback_data()
        
        return merged_df

    def _merge_station_data(self, station_files):
        """Merge station-level data files"""
        if not station_files:
            return pd.DataFrame()
        
        all_dfs = [file_info['data'] for file_info in station_files]
        
        # Get all unique columns
        all_columns = set()
        for df in all_dfs:
            all_columns.update(df.columns)
        
        print(f"   🏭 Merging {len(all_dfs)} station files with {len(all_columns)} unique columns")
        
        # Merge with column preservation
        merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        
        # Fill missing columns
        for col in all_columns:
            if col not in merged_df.columns:
                merged_df[col] = np.nan
        
        return merged_df

    def _merge_city_data(self, city_files):
        """Merge city-level data files"""
        if not city_files:
            return pd.DataFrame()
        
        all_dfs = [file_info['data'] for file_info in city_files]
        return pd.concat(all_dfs, ignore_index=True, sort=False)

    def _merge_with_other_data(self, main_df, other_files):
        """Merge main dataframe with other data files"""
        for file_info in other_files:
            other_df = file_info['data']
            # Try to merge based on common columns
            common_cols = set(main_df.columns) & set(other_df.columns)
            if common_cols:
                try:
                    main_df = pd.merge(main_df, other_df, on=list(common_cols), how='left', suffixes=('', '_other'))
                    print(f"   🔗 Merged with {file_info['filename']} using {len(common_cols)} common columns")
                except Exception as e:
                    print(f"   ⚠️  Could not merge with {file_info['filename']}: {e}")
        
        return main_df

    def _print_dataset_summary(self, dataset_files_summary):
        """Print comprehensive summary of loaded datasets"""
        print("\n" + "="*60)
        print("📊 DATASET LOADING SUMMARY")
        print("="*60)
        
        total_files = 0
        total_records = 0
        
        for dataset_name, files in dataset_files_summary.items():
            print(f"\n🏷️  {dataset_name}:")
            dataset_records = 0
            for file_info in files:
                record_count = len(file_info['data'])
                dataset_records += record_count
                total_records += record_count
                total_files += 1
                print(f"   📄 {file_info['filename']} ({file_info['file_type']}): {record_count:,} records")
            
            print(f"   📈 Total: {len(files)} files, {dataset_records:,} records")
        
        print(f"\n📈 GRAND TOTAL: {total_files} files, {total_records:,} records")
        print("="*60)

    def _apply_data_enhancements(self, df):
        """Apply data enhancements and filters"""
        print("\n🔧 Applying data enhancements...")
        
        if df.empty:
            print("⚠️  No data to enhance")
            return df
        
        initial_count = len(df)
        print(f"   📊 Initial records: {initial_count:,}")
        
        # Filter for Indian cities - but be more flexible
        if 'City' in df.columns:
            # Convert city names to lowercase for case-insensitive matching
            df['City_Lower'] = df['City'].str.lower()
            INDIAN_CITIES_LOWER = [city.lower() for city in INDIAN_CITIES]
            
            # Keep records that match Indian cities OR have no city data
            indian_city_mask = df['City_Lower'].isin(INDIAN_CITIES_LOWER)
            unknown_city_mask = df['City'].isna() | (df['City'] == '') | (df['City'] == 'Unknown')
            
            df = df[indian_city_mask | unknown_city_mask]
            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                print(f"   🏙️  Filtered {filtered_count:,} non-Indian city records")
            
            # Remove the temporary column
            df = df.drop('City_Lower', axis=1)
        
        # Filter date range more carefully
        if 'Date' in df.columns:
            # Ensure Date is datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Remove records with invalid dates but keep those without dates
            invalid_dates = df['Date'].isna()
            if invalid_dates.any():
                print(f"   ⚠️  Found {invalid_dates.sum():,} records with invalid dates (keeping them)")
            
            # Filter by date range but keep records without dates
            date_mask = (df['Date'] >= '2015-01-01') & (df['Date'] <= '2024-12-31')
            no_date_mask = df['Date'].isna()
            
            df = df[date_mask | no_date_mask]
            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                print(f"   📅 Filtered {filtered_count:,} records outside 2015-2024 range")
        
        # Remove duplicates more carefully
        initial_count_before_dedup = len(df)
        if initial_count_before_dedup > 0:
            # Build duplicate columns based on available columns
            dup_cols = []
            if 'City' in df.columns:
                dup_cols.append('City')
            if 'Date' in df.columns:
                dup_cols.append('Date')
            if 'Station' in df.columns:
                dup_cols.append('Station')
            if 'Datetime' in df.columns:
                dup_cols.append('Datetime')
            
            if dup_cols:
                df = df.drop_duplicates(subset=dup_cols, keep='first')
                removed_duplicates = initial_count_before_dedup - len(df)
                if removed_duplicates > 0:
                    print(f"   🔄 Removed {removed_duplicates:,} duplicate records")
            
            # Sort data
            sort_cols = []
            if 'Station' in df.columns:
                sort_cols.append('Station')
            if 'City' in df.columns:
                sort_cols.append('City')
            if 'Date' in df.columns:
                sort_cols.append('Date')
            if 'Datetime' in df.columns:
                sort_cols.append('Datetime')
            
            if sort_cols:
                df = df.sort_values(sort_cols)
        
        print(f"   ✅ Final records after enhancements: {len(df):,}")
        return df

    def _print_final_statistics(self, df):
        """Print final comprehensive statistics"""
        print("\n" + "="*60)
        print("📊 FINAL DATASET STATISTICS")
        print("="*60)
        
        print(f"📈 Total Records: {len(df):,}")
        
        if len(df) == 0:
            print("❌ No data available for statistics")
            print("="*60)
            return
        
        print(f"🏙️  Cities Covered: {df['City'].nunique() if 'City' in df.columns else 0}")
        print(f"🏭 Stations Covered: {df['Station'].nunique() if 'Station' in df.columns else 0}")
        
        if 'Date' in df.columns:
            # Handle potential NaT values safely
            valid_dates = df['Date'].dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                print(f"📅 Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                print(f"📅 Total Days: {(max_date - min_date).days} days")
            else:
                print("📅 Date Range: No valid dates available")
        
        # Data sources
        if 'Source_Dataset' in df.columns:
            print(f"📚 Datasets: {df['Source_Dataset'].nunique()}")
            print(f"📁 Files: {df['Source_File'].nunique()}")
            print(f"📄 File Types: {df['File_Type'].nunique()}")
        
        # Feature analysis
        print(f"🔧 Total Features: {len(df.columns)}")
        
        # Pollutant availability
        primary_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
        available_primary = [p for p in primary_pollutants if p in df.columns]
        print(f"🌫️  Primary Pollutants: {len(available_primary)}/{len(primary_pollutants)}")
        
        secondary_pollutants = ['NO', 'NOx', 'NH3', 'Benzene', 'Toluene', 'Xylene']
        available_secondary = [p for p in secondary_pollutants if p in df.columns]
        print(f"🧪 Secondary Pollutants: {len(available_secondary)}/{len(secondary_pollutants)}")
        
        meteorological = ['Temperature', 'Humidity', 'Pressure', 'Wind_Speed', 'Wind_Direction', 'Dew_Point', 'Precipitation']
        available_meteo = [m for m in meteorological if m in df.columns]
        print(f"🌤️  Meteorological Data: {len(available_meteo)}/{len(meteorological)}")
        
        # Data completeness
        print(f"\n📊 Data Completeness for Key Pollutants:")
        for col in primary_pollutants:
            if col in df.columns:
                non_null = df[col].notna().sum()
                completeness = (non_null / len(df)) * 100
                print(f"   {col}: {non_null:,} records ({completeness:.1f}%)")
        
        print("="*60)

    def _generate_comprehensive_fallback_data(self):
        """Generate comprehensive fallback data when no datasets are available"""
        print("🔄 Generating comprehensive fallback data...")
        
        # Create date range
        dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='D')
        
        fallback_data = []
        for city in INDIAN_CITIES[:10]:  # Limit to top 10 cities for performance
            for date in dates:
                # Base AQI with seasonal variation
                base_aqi = self._get_city_base_aqi(city)
                seasonal_factor = self._get_seasonal_factor(date)
                daily_variation = np.random.normal(0, 20)
                
                aqi = max(50, min(500, base_aqi * seasonal_factor + daily_variation))
                
                record = {
                    'City': city,
                    'Date': date,
                    'AQI': aqi,
                    'PM2.5': aqi * 0.48 + np.random.normal(0, 10),
                    'PM10': aqi * 0.77 + np.random.normal(0, 15),
                    'NO2': aqi * 0.28 + np.random.normal(0, 5),
                    'SO2': aqi * 0.22 + np.random.normal(0, 3),
                    'CO': aqi * 0.15 + np.random.normal(0, 2),
                    'O3': aqi * 0.10 + np.random.normal(0, 2),
                    'Temperature': np.random.normal(25, 5),
                    'Humidity': np.random.normal(60, 15),
                    'Source_Dataset': 'Fallback_Data',
                    'Source_File': 'generated_fallback.csv',
                    'File_Type': 'fallback'
                }
                
                fallback_data.append(record)
        
        df = pd.DataFrame(fallback_data)
        print(f"✅ Generated fallback data: {len(df):,} records")
        return df

    def _get_city_base_aqi(self, city):
        """Get base AQI for a city for fallback data generation"""
        base_aqis = {
            'Delhi': 280, 'Mumbai': 160, 'Bangalore': 120, 'Chennai': 140,
            'Kolkata': 220, 'Hyderabad': 130, 'Ahmedabad': 180, 'Pune': 110,
            'Surat': 150, 'Jaipur': 170, 'Lucknow': 260, 'Kanpur': 290,
            'Nagpur': 140, 'Indore': 130, 'Thane': 150, 'Bhopal': 120,
            'Visakhapatnam': 110, 'Patna': 240, 'Vadodara': 160, 'Ghaziabad': 270
        }
        return base_aqis.get(city, 150)

    def _get_seasonal_factor(self, date):
        """Get seasonal factor for fallback data generation"""
        month = date.month
        if month in [11, 12, 1, 2]:  # Winter
            return 1.4
        elif month in [3, 4]:  # Spring
            return 1.2
        elif month in [5, 6, 7]:  # Summer
            return 0.9
        else:  # Monsoon
            return 0.8

    def get_station_details(self, city=None):
        """Get station details for a specific city or all cities"""
        if self.stations_metadata is None or self.stations_metadata.empty:
            return []
        
        if city:
            stations = self.stations_metadata[self.stations_metadata['City'].str.lower() == city.lower()]
            return stations.to_dict('records')
        else:
            return self.stations_metadata.to_dict('records')

# Update the global instance
aqi_data_loader = ComprehensiveAQIDataLoader()

if __name__ == "__main__":
    print("🧪 Testing Comprehensive Data Loader...")
    loader = ComprehensiveAQIDataLoader()
    data = loader.load_historical_data()
    
    if not data.empty:
        print(f"\n✅ Successfully loaded comprehensive dataset with {len(data):,} records")
        print(f"📊 Available columns: {list(data.columns)}")
        
        # Test station metadata
        if loader.stations_metadata is not None:
            print(f"📍 Station metadata: {len(loader.stations_metadata)} stations")
            print(f"🏙️  Cities in station metadata: {loader.stations_metadata['City'].nunique()}")
    else:
        print("❌ Failed to load any data")

   