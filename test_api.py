# test_api.py
# Run in a NEW terminal while api.py is running:
# python test_api.py

import requests
import json

BASE = "http://localhost:5000"

print("=" * 55)
print("Testing Flight Segmentation API")
print("=" * 55)

# ── Test 1: Health check ──────────────────────────────────
print("\n--- Test 1: Health Check ---")
r = requests.get(f"{BASE}/health")
print(f"Status : {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 2: Champion customer ─────────────────────────────
print("\n--- Test 2: Champion Customer ---")
champion = {
    "L": 2500,    # long membership — 2500 days
    "R": 15,      # flew 15 days ago — very recent
    "F": 180,     # 180 total flights — very frequent
    "M": 420000,  # 420,000 km — huge distance
    "C": 0.95     # pays close to full fare
}
r = requests.post(f"{BASE}/predict", json=champion)
print(f"Status  : {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 3: At-Risk customer ──────────────────────────────
print("\n--- Test 3: At-Risk Customer ---")
at_risk = {
    "L": 2000,   # long member — 2000 days
    "R": 380,    # hasn't flown in 380 days
    "F": 8,      # only 8 total flights
    "M": 22000,  # low km
    "C": 0.71    # moderate discount
}
r = requests.post(f"{BASE}/predict", json=at_risk)
print(f"Status  : {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 4: Occasional flyer ──────────────────────────────
print("\n--- Test 4: Occasional Flyer ---")
occasional = {
    "L": 400,    # new member
    "R": 200,    # hasn't flown recently
    "F": 4,      # very few flights
    "M": 9000,   # low km
    "C": 0.65    # price sensitive
}
r = requests.post(f"{BASE}/predict", json=occasional)
print(f"Status  : {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 5: Segments list ─────────────────────────────────
print("\n--- Test 5: All Segments ---")
r = requests.get(f"{BASE}/segments")
print(f"Status  : {r.status_code}")
data = r.json()
print(f"Total segments: {data['total_segments']}")
for seg in data['segments']:
    print(f"  Cluster {seg['cluster_id']}: {seg['name']} ({seg['share']})")

print("\n" + "=" * 55)
print("All tests complete")