"""
Simple backend API tester for the AQI project.
Usage:
  python test_backend.py --base http://127.0.0.1:5000

This script calls key backend endpoints and prints nicely formatted responses.
"""
import requests
import argparse
import json
import sys
from datetime import datetime

DEFAULT_BASE = "http://127.0.0.1:5000"


def pretty_print(title, data):
    print('\n' + '=' * 80)
    print(title)
    print('-' * 80)
    try:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(data)


def call_health(base):
    url = f"{base}/api/health"
    r = requests.get(url, timeout=10)
    return r.status_code, r.json() if r.content else None


def call_status(base):
    url = f"{base}/api/status"
    r = requests.get(url, timeout=10)
    return r.status_code, r.json() if r.content else None


def call_cities(base):
    url = f"{base}/api/cities"
    r = requests.get(url, timeout=10)
    return r.status_code, r.json() if r.content else None


def call_realtime(base, city):
    url = f"{base}/api/realtime/{requests.utils.requote_uri(city)}"
    r = requests.get(url, timeout=15)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


def call_predict(base, city):
    url = f"{base}/api/predict"
    payload = {
        "city": city,
        "date": datetime.now().strftime('%Y-%m-%d'),
        "model_type": "auto"
    }
    r = requests.post(url, json=payload, timeout=20)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


def call_initialize(base):
    url = f"{base}/api/initialize"
    r = requests.post(url, timeout=30)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', '-b', default=DEFAULT_BASE, help='Backend base URL (http://host:port)')
    parser.add_argument('--city', '-c', default='Delhi', help='City to test realtime & predict endpoints')
    parser.add_argument('--skip-init', action='store_true', help='Skip calling /api/initialize')
    args = parser.parse_args()

    base = args.base.rstrip('/')
    city = args.city

    print(f"Testing backend at: {base}")

    code, data = call_health(base)
    pretty_print(f"GET /api/health -> {code}", data)

    code, data = call_status(base)
    pretty_print(f"GET /api/status -> {code}", data)

    if not args.skip_init:
        print('\nCalling POST /api/initialize (this may take a while)')
        code, data = call_initialize(base)
        pretty_print(f"POST /api/initialize -> {code}", data)

    code, data = call_cities(base)
    pretty_print(f"GET /api/cities -> {code}", data)

    code, data = call_realtime(base, city)
    pretty_print(f"GET /api/realtime/{city} -> {code}", data)

    code, data = call_predict(base, city)
    pretty_print(f"POST /api/predict (city={city}) -> {code}", data)

    print('\nAll tests completed.')
