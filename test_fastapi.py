# test_fastapi.py
import requests
import sys

BASE = "http://localhost:8000"

print("Testing FastAPI...")

# Test 1: Health
r = requests.get(f"{BASE}/")
assert r.status_code == 200, f"Health failed: {r.status_code}"
print("Health check PASSED")

# Test 2: Prediction
r = requests.post(f"{BASE}/predict", json={
    "L": 2500, "R": 15, "F": 180, "M": 420000, "C": 0.95
})
assert r.status_code == 200, f"Predict failed: {r.status_code}"
print(f"Predict PASSED: {r.json()['segment_name']}")

# Test 3: Validation
r = requests.post(f"{BASE}/predict", json={"L": 2500})
assert r.status_code == 422, f"Validation failed: {r.status_code}"
print("Validation PASSED")

print("All tests passed!")