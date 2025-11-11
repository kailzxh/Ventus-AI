#!/usr/bin/env python
import argparse
import requests
import sys
import json
from datetime import datetime, timedelta

def test_prediction_endpoint(base_url, city):
    """Test the prediction endpoint"""
    print(f"\n🔍 Testing prediction endpoint for {city}...")
    
    try:
        # Test single day prediction
        url = f"{base_url}/predict"
        data = {"city": city, "days": 1}
        
        print(f"📡 POST {url}")
        print(f"📦 Data: {json.dumps(data)}")
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ Got prediction: {json.dumps(result, indent=2)}")
        
        # Validate response structure
        if not isinstance(result, list):
            print("❌ Error: Expected list response")
            return False
        
        for prediction in result:
            if not all(k in prediction for k in ['city', 'date', 'predicted_aqi', 'category']):
                print(f"❌ Error: Missing required fields in prediction: {prediction}")
                return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling prediction endpoint: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_realtime_endpoint(base_url, city):
    """Test the realtime endpoint"""
    print(f"\n🔍 Testing realtime endpoint for {city}...")
    
    try:
        url = f"{base_url}/realtime/{city}"
        print(f"📡 GET {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ Got realtime data: {json.dumps(result, indent=2)}")
        
        # Validate response structure
        if not isinstance(result, dict):
            print("❌ Error: Expected dictionary response")
            return False
            
        required_fields = ['city', 'aqi', 'category', 'timestamp']
        if not all(k in result for k in required_fields):
            print(f"❌ Error: Missing required fields in response: {result}")
            return False
            
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling realtime endpoint: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_stations_endpoint(base_url, city):
    """Test the stations endpoint"""
    print(f"\n🔍 Testing stations endpoint for {city}...")
    
    try:
        url = f"{base_url}/stations/{city}"
        print(f"📡 GET {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ Got stations data: {json.dumps(result, indent=2)}")
        
        # Validate response structure
        if not isinstance(result, list):
            print("❌ Error: Expected list response")
            return False
            
        for station in result:
            if not all(k in station for k in ['station_id', 'station_name']):
                print(f"❌ Error: Missing required fields in station: {station}")
                return False
                
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling stations endpoint: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test AQI Backend API')
    parser.add_argument('--base', help='Base URL of the API server', required=True)
    parser.add_argument('--city', help='City to test predictions for', required=True)
    parser.add_argument('--skip-init', help='Skip initialization check', action='store_true')
    args = parser.parse_args()

    base_url = args.base.rstrip('/')
    print(f"🚀 Testing AQI Backend API at {base_url}")

    success = True
    
    # Test prediction endpoint
    if not test_prediction_endpoint(base_url, args.city):
        success = False
        
    # Test realtime endpoint
    if not test_realtime_endpoint(base_url, args.city):
        success = False
        
    # Test stations endpoint
    if not test_stations_endpoint(base_url, args.city):
        success = False
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()