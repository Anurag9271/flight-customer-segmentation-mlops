# verify_predictions.py
import requests
import json

BASE = "http://localhost:5000"

test_customers = [
    {"L": 2500, "R": 15,  "F": 180, "M": 420000, "C": 0.95},
    {"L": 2000, "R": 380, "F": 8,   "M": 22000,  "C": 0.71},
    {"L": 400,  "R": 200, "F": 4,   "M": 9000,   "C": 0.65},
]

print("Verifying predictions from container...")
print("=" * 50)

for i, customer in enumerate(test_customers, 1):
    response = requests.post(f"{BASE}/predict", json=customer)
    result   = response.json()
    print(f"\nCustomer {i}:")
    print(f"  Input   : {customer}")
    print(f"  Segment : {result.get('segment')}")
    print(f"  Cluster : {result.get('cluster_id')}")
    print(f"  Status  : {response.status_code}")

print("\n" + "=" * 50)
print("Verification complete")