#!/usr/bin/env bash
# Simple wrapper to run the Python API tests
# Usage: ./run_tests.sh http://192.168.1.12:5000

BASE=${1:-http://127.0.0.1:5000}
CITY=${2:-Delhi}

python3 test_backend.py --base "$BASE" --city "$CITY" --skip-init
