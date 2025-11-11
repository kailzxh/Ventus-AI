# backend/data_loader.py
import pandas as pd
import numpy as np
import kaggle
import os
import requests
import json
import glob
from datetime import datetime, timedelta
# *** IMPORT THE NEW MAPS ***
from config import Config, INDIAN_CITIES, CITY_NORMALIZATION_MAP, NORMALIZED_CITIES

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
            # *** Show normalized cities from metadata ***
            # self.stations_metadata = self._normalize_city_names(self.stations_metadata) # This is now done in _extract
            cities = self.stations_metadata['City'].dropna().unique()
            print(f"🏙️  Normalized Cities in metadata: {len(cities)} - {list(cities)[:10]}...")

    def load_historical_data(self):
        """Load ALL historical AQI data from multiple Kaggle datasets for 2015-2024"""
        print("📥 Loading Comprehensive Historical AQI Data (2015-2024)...")
        
        # Set Kaggle credentials
        os.environ['KAGGLE_USERNAME'] = 'kailzxh'
        os.environ['KAGGLE_KEY'] = 'a0601d99ed2c238727c9cc1ba4bb562c'
        
        # Define datasets
        datasets_to_load = [
            {
                'name': 'Indian AQI 2015-2020',
                'id': 'rohanrao/air-quality-data-in-india',
                'download_name': 'india-aqi-2015-2020',
            },
            {
                'name': 'Extended Indian AQI 2015-2024',
                'id': 'ankushpanday1/air-quality-data-in-india-2015-2024',
                'download_name': 'india-aqi-2015-2024', 
            }
        ]
        
        all_loaded_data = []
        dataset_files_summary = {}
        
        for dataset_info in datasets_to_load:
            try:
                print(f"\n🔍 Downloading: {dataset_info['name']}")
                print(f"   Dataset ID: {dataset_info['id']}")
                
                download_path = f"./data/raw/{dataset_info['download_name']}"
                os.makedirs(download_path, exist_ok=True)
                
                kaggle.api.dataset_download_files(
                    dataset_info['id'],
                    path=download_path,
                    unzip=True,
                    quiet=True # Set to True to reduce noise
                )
                
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
        
        self._print_dataset_summary(dataset_files_summary)
        
        # *** Store station metadata *before* merging ***
        self._extract_station_metadata(all_loaded_data)
        
        if all_loaded_data:
            combined_df = self._intelligently_merge_datasets(all_loaded_data)
            print(f"\n✅ Final Combined Dataset: {len(combined_df):,} records")
        else:
            print("🔄 No datasets loaded, generating fallback data...")
            combined_df = self._generate_comprehensive_fallback_data()
        
        combined_df = self._apply_data_enhancements(combined_df)
        
        self._debug_station_mapping()
        self._print_final_statistics(combined_df)
        return combined_df

    def _normalize_city_names(self, df):
        """Applies the master CITY_NORMALIZATION_MAP to the 'City' column."""
        if 'City' not in df.columns:
            print(" ⚠️ No 'City' column found to normalize.")
            df['City'] = np.nan # Create it if it doesn't exist
            return df

        print(" 🏙️ Normalizing city names...")
        initial_cities = df['City'].dropna().unique()
        
        lower_to_canonical_map = {k.lower(): v for k, v in CITY_NORMALIZATION_MAP.items()}
        
        if 'City_Original' not in df.columns:
            df['City_Original'] = df['City'] 
        
        df['City_Lower'] = df['City'].astype(str).str.lower().str.strip()
        df['City'] = df['City_Lower'].map(lower_to_canonical_map)
        
        unmapped = df[df['City'].isna() & df['City_Lower'].notna() & (df['City_Lower'] != 'nan') & (df['City_Lower'] != '') & (df['City_Lower'] != 'unknown')]['City_Original'].unique()
        if len(unmapped) > 0:
            print(f" ⚠️ Found {len(unmapped)} unmapped cities. Examples: {list(unmapped)[:10]}")
            
        df['City'] = df['City'].fillna(np.nan) 
        
        final_cities = df['City'].dropna().unique()
        print(f" ✅ City normalization complete. {len(initial_cities)} original -> {len(final_cities)} canonical cities.")
        
        df = df.drop(columns=['City_Lower'])
        return df

    def _extract_station_metadata(self, all_loaded_data):
        """Extract and store station metadata separately with better merging"""
        print("\n📍 Extracting station metadata...")
        station_metadata_dfs = []
        
        for file_info in all_loaded_data:
            if file_info['file_type'] == 'station_metadata':
                df = file_info['data']
                if len(df) > 0 and ('StationId' in df.columns or 'StationName' in df.columns or 'Station' in df.columns):
                    station_metadata_dfs.append(df)
                    print(f"   ✅ Found {len(df)} station records from {file_info['filename']}")
        
        if station_metadata_dfs:
            self.stations_metadata = pd.concat(station_metadata_dfs, ignore_index=True, sort=False)
            
            if 'StationId' in self.stations_metadata.columns:
                self.stations_metadata = self.stations_metadata.drop_duplicates(subset=['StationId'], keep='first')
            elif 'StationName' in self.stations_metadata.columns:
                self.stations_metadata = self.stations_metadata.drop_duplicates(subset=['StationName'], keep='first')
            elif 'Station' in self.stations_metadata.columns:
                self.stations_metadata = self.stations_metadata.drop_duplicates(subset=['Station'], keep='first')
            
            if 'StationId' not in self.stations_metadata.columns and 'Station' in self.stations_metadata.columns:
                self.stations_metadata['StationId'] = self.stations_metadata['Station']
            if 'StationName' not in self.stations_metadata.columns and 'Station' in self.stations_metadata.columns:
                self.stations_metadata['StationName'] = self.stations_metadata['Station']
            
            self.stations_metadata = self._normalize_city_names(self.stations_metadata)
            print(f"✅ Loaded station metadata for {len(self.stations_metadata)} unique stations")
            
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
                station_cols = [col for col in ['StationId', 'Station', 'StationName', 'City', 'State', 'Latitude', 'Longitude'] if col in df.columns]
                
                if station_cols:
                    station_info = df[station_cols].drop_duplicates()
                    station_data.append(station_info)
                    print(f"       📊 Extracted {len(station_info)} stations from {file_info['filename']}")
        
        if station_data:
            combined_stations = pd.concat(station_data, ignore_index=True)
            
            if 'StationId' in combined_stations.columns:
                combined_stations = combined_stations.drop_duplicates(subset=['StationId'], keep='first')
            elif 'Station' in combined_stations.columns:
                combined_stations = combined_stations.drop_duplicates(subset=['Station'], keep='first')
            
            if 'StationName' not in combined_stations.columns:
                if 'Station' in combined_stations.columns:
                    combined_stations['StationName'] = combined_stations['Station']
                elif 'StationId' in combined_stations.columns:
                    combined_stations['StationName'] = combined_stations['StationId']
            
            # *** Normalize cities here too ***
            combined_stations = self._normalize_city_names(combined_stations)
            
            print(f"   ✅ Created enhanced metadata for {len(combined_stations)} stations from data")
            return combined_stations
        else:
            print("   ⚠️  Could not create station metadata from data")
            return pd.DataFrame()

    def _load_all_files_from_dataset(self, download_path, dataset_info):
        """Load all files from a dataset directory with specific handling for each file type"""
        print(f"   📂 Scanning directory: {download_path}")
        loaded_files = []
        
        if not os.path.exists(download_path):
            print(f"   ❌ Directory not found: {download_path}")
            return loaded_files
            
        csv_files = glob.glob(os.path.join(download_path, "**/*.csv"), recursive=True)
        print(f"   📊 Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            print(f"   🔍 Processing: {filename}")
            
            df = self._load_csv_file(csv_file)
            if df is not None and len(df) > 0:
                df = self._process_file_by_type(df, filename, dataset_info['name'])
                loaded_files.append({
                    'data': df,
                    'filename': filename,
                    'dataset': dataset_info['name'],
                    'file_type': self._classify_file_type(filename),
                    'file_path': csv_file
                })
                print(f"       ✅ Loaded {filename}: {len(df):,} records, {len(df.columns)} columns")
            else:
                print(f"       ❌ Failed to load {filename}")
        
        return loaded_files

    def _classify_file_type(self, filename):
        """Classify files by their type for specialized processing"""
        filename_lower = filename.lower()
        
        if 'city_day' in filename_lower: return 'city_daily'
        elif 'city_hour' in filename_lower: return 'city_hourly' 
        elif 'station_day' in filename_lower: return 'station_daily'
        elif 'station_hour' in filename_lower: return 'station_hourly'
        elif 'stations' in filename_lower: return 'station_metadata'
        elif 'aqi_data' in filename_lower: return 'aqi_data'
        elif 'data' in filename_lower: return 'general_data'
        else: return 'unknown'

    def _process_file_by_type(self, df, filename, dataset_name):
        """Apply specialized processing based on file type with better station handling"""
        file_type = self._classify_file_type(filename)
        
        df['Source_Dataset'] = dataset_name
        df['Source_File'] = filename
        df['File_Type'] = file_type
        
        if file_type in ['city_daily', 'city_hourly', 'station_daily', 'station_hourly']:
            df = self._standardize_pollutant_data(df, file_type)
        elif file_type == 'station_metadata':
            df = self._standardize_station_metadata(df)
        elif file_type == 'aqi_data':
            df = self._standardize_aqi_data(df)
        
        df = self._apply_universal_standardization(df)
        return df

    def _enhance_station_data_with_metadata(self, df):
        """Enhance station data with proper station names from metadata"""
        if self.stations_metadata is None or self.stations_metadata.empty:
            return df
        
        # This function is now deferred to _intelligently_merge_datasets
        # We don't want to merge here, just standardize
        return df

    def _standardize_pollutant_data(self, df, file_type):
        """Standardize pollutant data files with enhanced station handling and name mapping"""
        print(f"       Standardizing {file_type} data...")
        
        column_mapping = {
            'station_id': 'StationId', 'StationId': 'StationId',
            'station': 'Station', 'Station': 'Station', 'site': 'Station', 'location': 'Station',
            'station_name': 'StationName', 'StationName': 'StationName',
            'city': 'City', 'City': 'City',
            'date': 'Date', 'Date': 'Date',
            'datetime': 'Datetime', 'Datetime': 'Datetime',
            'pm25': 'PM2.5', 'PM2.5': 'PM2.5', 'pm2.5': 'PM2.5',
            'pm10': 'PM10', 'PM10': 'PM10',
            'no2': 'NO2', 'NO2': 'NO2',
            'so2': 'SO2', 'SO2': 'SO2',
            'co': 'CO', 'CO': 'CO',
            'o3': 'O3', 'O3': 'O3',
            'aqi': 'AQI', 'AQI': 'AQI',
            'aqi_bucket': 'AQI_Bucket', 'AQI_Bucket': 'AQI_Bucket',
            'temperature': 'Temperature', 'Temperature': 'Temperature',
            'humidity': 'Humidity', 'Humidity': 'Humidity',
            'pressure': 'Pressure', 'Pressure': 'Pressure',
            'wind_speed': 'Wind_Speed', 'Wind_Speed': 'Wind_Speed',
            'state': 'State', 'State': 'State',
            'latitude': 'Latitude', 'Latitude': 'Latitude',
            'longitude': 'Longitude', 'Longitude': 'Longitude',
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        if 'StationId' not in df.columns and 'Station' in df.columns:
            station_values = df['Station'].dropna().unique()
            has_station_ids = any(len(str(val)) <= 10 and any(c.isdigit() for c in str(val)) for val in station_values)
            if has_station_ids:
                df['StationId'] = df['Station']
            else:
                df['StationName'] = df['Station']
        
        if 'Station' not in df.columns:
            if 'StationId' in df.columns:
                df['Station'] = df['StationId']
            elif 'StationName' in df.columns:
                df['Station'] = df['StationName']
            elif 'City' in df.columns:
                df['Station'] = df['City'] + '_Central'
                print(f"       🔧 Created synthetic station names for city-level data")
        
        return df

    def _standardize_station_metadata(self, df):
        """Standardize station metadata files with comprehensive mapping"""
        print("       Standardizing station metadata...")
        column_mapping = {
            'station_id': 'StationId', 'StationId': 'StationId',
            'station': 'Station', 'Station': 'Station',
            'name': 'StationName', 'station_name': 'StationName', 'Site': 'StationName',
            'city': 'City', 'City': 'City',
            'state': 'State', 'State': 'State',
            'latitude': 'Latitude', 'Latitude': 'Latitude',
            'longitude': 'Longitude', 'Longitude': 'Longitude',
            'type': 'Station_Type', 'station_type': 'Station_Type',
            'status': 'Status', 'Status': 'Status',
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        if 'StationId' not in df.columns and 'Station' in df.columns:
            df['StationId'] = df['Station']
        if 'StationName' not in df.columns:
            if 'Station' in df.columns:
                df['StationName'] = df['Station']
            elif 'StationId' in df.columns:
                df['StationName'] = df['StationId']
        
        if 'StationName' in df.columns:
            df['StationName'] = df['StationName'].astype(str).str.strip().replace(['Unknown', 'unknown', ''], np.nan)
        if 'City' in df.columns:
            df['City'] = df['City'].astype(str).str.strip().replace(['Unknown', 'unknown', ''], np.nan)
        
        print(f"       ✅ Standardized {len(df)} station metadata records")
        return df

    def _standardize_aqi_data(self, df):
        """Standardize AQI-specific data files"""
        print("       Standardizing AQI data...")
        return self._standardize_pollutant_data(df, 'aqi_data')

    def _apply_universal_standardization(self, df):
        """Apply universal standardization to all dataframes"""
        
        # *** FIX for Failure A: Create 'Date' from 'Datetime' ***
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
            if 'Date' not in df.columns:
                df['Date'] = pd.to_datetime(df['Datetime'].dt.date)
                print(f"       🔧 Created 'Date' column from 'Datetime'")
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        numeric_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI', 'Temperature', 'Humidity', 'Pressure',
                           'Wind_Speed', 'Latitude', 'Longitude', 'NO', 'NOx', 'NH3', 'Benzene', 'Toluene', 'Xylene']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def _load_csv_file(self, file_path):
        """Load CSV file with multiple encoding attempts"""
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                if len(df) > 0:
                    return df
            except Exception:
                continue
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', low_memory=False)
            return df if len(df) > 0 else None
        except Exception as e:
            print(f"       ❌ Failed to load {file_path}: {e}")
            return None

    # ***
    # *** RE-WRITTEN: _intelligently_merge_datasets (Fixes Failure B) ***
    # ***
    def _intelligently_merge_datasets(self, all_loaded_data):
        """Intelligently merge all datasets preserving all features with station focus"""
        print("\n🔄 Intelligently merging datasets...")
        
        if not all_loaded_data:
            print("⚠️  No datasets to merge")
            return pd.DataFrame()
        
        station_files = [f['data'] for f in all_loaded_data if f['file_type'] in ['station_daily', 'station_hourly']]
        city_files = [f['data'] for f in all_loaded_data if f['file_type'] in ['city_daily', 'city_hourly']]
        
        print(f"   📊 Found: {len(station_files)} station files, {len(city_files)} city files")
        
        # Combine all station data and all city data
        merged_station_df = pd.DataFrame()
        if station_files:
            merged_station_df = pd.concat(station_files, ignore_index=True, sort=False)
            print(f"   🏭 Merged {len(station_files)} station files into {len(merged_station_df)} records")

        merged_city_df = pd.DataFrame()
        if city_files:
            merged_city_df = pd.concat(city_files, ignore_index=True, sort=False)
            print(f"   🏙️  Merged {len(city_files)} city files into {len(merged_city_df)} records")

        # *** FIX for Failure B: Merge station data with metadata to get City ***
        if not merged_station_df.empty and self.stations_metadata is not None:
            print(f"   🔗 Merging {len(merged_station_df)} station records with {len(self.stations_metadata)} metadata records...")
            
            meta_cols = ['StationId', 'City', 'State']
            if 'StationName' in self.stations_metadata.columns:
                 meta_cols.append('StationName')
            
            meta_to_merge = self.stations_metadata[meta_cols].drop_duplicates(subset=['StationId'])
            
            merged_station_df = pd.merge(
                merged_station_df,
                meta_to_merge,
                on='StationId',
                how='left',
                suffixes=('', '_meta')
            )
            
            # Fill main 'City' column from metadata
            if 'City_meta' in merged_station_df.columns:
                if 'City' not in merged_station_df.columns:
                     merged_station_df['City'] = merged_station_df['City_meta']
                else:
                     merged_station_df['City'] = merged_station_df['City'].fillna(merged_station_df['City_meta'])
                
                merged_station_df = merged_station_df.drop(columns=['City_meta'])
                
            city_fill_count = merged_station_df['City'].notna().sum()
            print(f"   ✅ Merge complete. {city_fill_count} station records now have city info.")
        
        # Now combine the (enhanced) station data with the city data
        combined_df = pd.concat([merged_station_df, merged_city_df], ignore_index=True, sort=False)
        
        if combined_df.empty:
            print("   ⚠️  All merge attempts failed, generating fallback data")
            return self._generate_comprehensive_fallback_data()
        
        return combined_df

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

    # ***
    # *** RE-WRITTEN: _apply_data_enhancements (Fixes Failure A & adds Aggregation) ***
    # ***
    def _apply_data_enhancements(self, df):
        """Apply data enhancements, aggregation, and filters"""
        print("\n🔧 Applying data enhancements...")
        
        if df.empty:
            print("⚠️  No data to enhance")
            return df
        
        initial_count = len(df)
        print(f"   📊 Initial records: {initial_count:,}")

        # *** STEP 1: NORMALIZE CITY NAMES (from metadata merge) ***
        # This will fill any remaining NaNs from city-level files
        df = self._normalize_city_names(df)
        
        # *** STEP 2: AGGREGATE TO DAILY DATA ***
        # This fixes the hourly vs daily problem and prepares data for all models
        print("   ...Aggregating all data to daily averages...")
        
        # Define columns to aggregate (numeric)
        agg_cols = [
            'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 
            'Toluene', 'Xylene', 'AQI', 'Temperature', 'Humidity', 'Pressure', 'Wind_Speed'
        ]
        agg_cols_present = [col for col in agg_cols if col in df.columns]
        
        # Define columns to group by
        group_by_cols = ['City', 'Station', 'Date']
        
        # Keep non-numeric columns by taking the 'first'
        first_cols = [
            'StationId', 'StationName', 'AQI_Bucket', 'Source_Dataset', 'City_Original'
        ]
        first_cols_present = [col for col in first_cols if col in df.columns]
        
        agg_dict = {col: 'mean' for col in agg_cols_present}
        for col in first_cols_present:
            agg_dict[col] = 'first'
        
        # Filter out rows with no Date (e.g., from metadata files)
        df = df.dropna(subset=['Date'])
        
        df_agg = df.groupby(group_by_cols).agg(agg_dict).reset_index()
        print(f"   ...Aggregated {len(df)} records into {len(df_agg)} daily records.")
        df = df_agg
        initial_count = len(df) # Reset count after aggregation
        
        # *** STEP 3: FILTER FOR VALID CITIES ***
        if 'City' in df.columns:
            # Filter based on the *normalized* city list
            # We no longer keep 'Unknown' cities, as all data should be mapped
            indian_city_mask = df['City'].isin(NORMALIZED_CITIES)
            
            df = df[indian_city_mask]
            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                print(f"   🏙️  Filtered {filtered_count:,} unmapped/non-Indian city records")
            
            initial_count = len(df)

        # *** STEP 4: FILTER DATE RANGE ***
        if 'Date' in df.columns:
            # We already dropped NaT dates during aggregation
            date_mask = (df['Date'] >= '2015-01-01') & (df['Date'] <= '2024-12-31')
            df = df[date_mask]
            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                print(f"   📅 Filtered {filtered_count:,} records outside 2015-2024 range")
            
            initial_count = len(df)

        # *** STEP 5: REMOVE DUPLICATES (Post-Aggregation) ***
        # We drop duplicates based on City, Station, and Date
        # This is already handled by the groupby, but we do it again
        dup_cols = ['City', 'Date', 'Station']
        df = df.drop_duplicates(subset=dup_cols, keep='first')
        removed_duplicates = initial_count - len(df)
        if removed_duplicates > 0:
            print(f"   🔄 Removed {removed_duplicates:,} duplicate records")
        
        # Sort data
        df = df.sort_values(['City', 'Station', 'Date'])
        
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
        
        if 'City' in df.columns:
             print(f"🏙️  Cities Covered (Normalized): {df['City'].nunique()}")
        if 'City_Original' in df.columns:
             print(f"🏙️  Cities Found (Original): {df['City_Original'].nunique()}")
             
        print(f"🏭 Stations Covered: {df['Station'].nunique() if 'Station' in df.columns else 0}")
        
        if 'Date' in df.columns:
            valid_dates = df['Date'].dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                print(f"📅 Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                print(f"📅 Total Days: {(max_date - min_date).days} days")
            else:
                print("📅 Date Range: No valid dates available")
        
        if 'Source_Dataset' in df.columns:
            print(f"📚 Datasets: {df['Source_Dataset'].nunique()}")
        
        print(f"🔧 Total Features: {len(df.columns)}")
        
        primary_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
        available_primary = [p for p in primary_pollutants if p in df.columns]
        print(f"🌫️  Primary Pollutants: {len(available_primary)}/{len(primary_pollutants)}")
        
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
        dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='D')
        fallback_data = []
        for city in NORMALIZED_CITIES[:10]:
            for date in dates:
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
                    'Source_Dataset': 'Fallback_Data',
                }
                fallback_data.append(record)
        
        df = pd.DataFrame(fallback_data)
        print(f"✅ Generated fallback data: {len(df):,} records")
        return df

    def _get_city_base_aqi(self, city):
        """Get base AQI for a city for fallback data generation"""
        base_aqis = {
            'delhi': 280, 'mumbai': 160, 'bengaluru': 120, 'chennai': 140,
            'kolkata': 220, 'hyderabad': 130, 'ahmedabad': 180, 'pune': 110,
        }
        return base_aqis.get(city, 150)

    def _get_seasonal_factor(self, date):
        """Get seasonal factor for fallback data generation"""
        month = date.month
        if month in [11, 12, 1, 2]:  return 1.4
        elif month in [3, 4]:  return 1.2
        elif month in [5, 6, 7]:  return 0.9
        else:  return 0.8

    def get_station_details(self, city=None):
        """Get station details for a specific city or all cities"""
        if self.stations_metadata is None or self.stations_metadata.empty:
            return []
        
        if city:
            normalized_city = CITY_NORMALIZATION_MAP.get(city.lower(), city.lower())
            stations = self.stations_metadata[self.stations_metadata['City'] == normalized_city]
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
        
        if loader.stations_metadata is not None:
            print(f"📍 Station metadata: {len(loader.stations_metadata)} stations")
            print(f"🏙️  Cities in station metadata: {loader.stations_metadata['City'].nunique()}")
            
        print("\n🧪 Testing City Normalization:")
        if 'City' in data.columns:
            print(f"Normalized cities in final data: {data['City'].dropna().unique()}")
            if 'Delhi' in data['City'].unique():
                 print("❌ FAILED: 'Delhi' (capitalized) still exists.")
            if 'NCT' in data['City'].unique():
                 print("❌ FAILED: 'NCT' still exists.")
            if 'delhi' in data['City'].unique():
                 print("✅ SUCCESS: 'delhi' (normalized) exists.")
        if 'City_Original' in data.columns:
             print(f"Original cities found: {data['City_Original'].dropna().unique()[:20]}...")
    else:
        print("❌ Failed to load any data")