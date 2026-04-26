# test_e2e.py
# ─────────────────────────────────────────────────────────
# End-to-End Integration Test
# Day 10 — Verifies full pipeline consistency:
#   Direct Model → Flask API → Docker Container
#
# Run: python test_e2e.py
# Requirements: api.py must be running on port 5000
# ─────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import pandas as pd
import joblib
import requests
import json

# ── Segment mapping ───────────────────────────────────────
SEGMENT_MAP = {
    0: "Loyal Regulars",
    1: "Occasional Leisure Flyers",
    2: "Champions",
    3: "At-Risk Customers"
}

# ── Test customers covering all 4 segments ────────────────
TEST_CUSTOMERS = [
    {
        "name"      : "Champion Customer",
        "input"     : {"L": 2500, "R": 15,  "F": 180, "M": 420000, "C": 0.95},
        "expected"  : 2
    },
    {
        "name"      : "At-Risk Customer",
        "input"     : {"L": 2000, "R": 380, "F": 8,   "M": 22000,  "C": 0.71},
        "expected"  : 3
    },
    {
        "name"      : "Occasional Flyer",
        "input"     : {"L": 400,  "R": 200, "F": 4,   "M": 9000,   "C": 0.65},
        "expected"  : 1
    },
    {
        "name"      : "Loyal Regular",
        "input"     : {"L": 300, "R": 45,  "F": 40,  "M": 85000,  "C": 0.82},
        "expected"  : 0
    },
]

# ─────────────────────────────────────────────────────────
# LAYER 1 — Direct Model Prediction
# Load pkl files directly and predict without Flask
# This is the ground truth baseline
# ─────────────────────────────────────────────────────────
def predict_direct(customer_input):
    """
    Make prediction by loading model directly.
    This is the baseline — what the model actually predicts.
    """
    BASE      = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE, "outputs", "models")

    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    km_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))

    L = float(customer_input['L'])
    R = float(customer_input['R'])
    F = float(customer_input['F'])
    M = float(customer_input['M'])
    C = float(customer_input['C'])

    # Apply same log transform as training
    features = pd.DataFrame(
        [[np.log1p(L), np.log1p(R), np.log1p(F), np.log1p(M), C]],
        columns=['L', 'R', 'F', 'M', 'C']
    )

    scaled     = scaler.transform(features)
    cluster_id = int(km_model.predict(scaled)[0])

    return cluster_id


# ─────────────────────────────────────────────────────────
# LAYER 2 & 3 — API Prediction
# Send request to Flask API (local or Docker)
# Both use port 5000 — Docker port maps to same port
# ─────────────────────────────────────────────────────────
def predict_via_api(customer_input, base_url="http://localhost:5000"):
    """
    Make prediction by calling Flask API.
    Works for both local api.py and Dockerized container
    because both expose port 5000.
    """
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=customer_input,
            timeout=10
        )

        if response.status_code == 200:
            return response.json().get("cluster_id")
        else:
            print(f"    API Error {response.status_code}: {response.json()}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"    Cannot connect to {base_url}")
        print("    Make sure Flask API is running")
        return None


# ─────────────────────────────────────────────────────────
# RUN TESTS — Compare all three layers
# ─────────────────────────────────────────────────────────
def run_e2e_tests(api_url="http://localhost:5000"):

    print("\n" + "=" * 60)
    print("  END-TO-END INTEGRATION TEST")
    print("  Flight Customer Segmentation")
    print("=" * 60)
    print(f"  API URL: {api_url}")
    print("=" * 60)

    results = []
    all_passed = True

    for customer in TEST_CUSTOMERS:

        print(f"\n  Customer : {customer['name']}")
        print(f"  Input    : {customer['input']}")

        # Layer 1 — Direct model
        direct_cluster = predict_direct(customer['input'])
        direct_segment = SEGMENT_MAP.get(direct_cluster, "Unknown")
        print(f"  Direct   : Cluster {direct_cluster} → {direct_segment}")

        # Layer 2/3 — Via API
        api_cluster = predict_via_api(customer['input'], api_url)
        if api_cluster is not None:
            api_segment = SEGMENT_MAP.get(api_cluster, "Unknown")
            print(f"  API      : Cluster {api_cluster} → {api_segment}")
        else:
            api_segment = "ERROR"
            print(f"  API      : ERROR — could not get prediction")

        # Compare results
        match = (direct_cluster == api_cluster)

        if match:
            print(f"  Result   : ✓ MATCH — predictions consistent")
        else:
            print(f"  Result   : ✗ MISMATCH — predictions differ!")
            all_passed = False

        results.append({
            "customer"      : customer['name'],
            "input"         : customer['input'],
            "direct_cluster": direct_cluster,
            "api_cluster"   : api_cluster,
            "match"         : match
        })

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r['match'])
    total  = len(results)

    print(f"  RESULTS: {passed}/{total} tests passed")

    if all_passed:
        print("  STATUS : ✓ ALL TESTS PASSED")
        print("  The complete pipeline is consistent end-to-end")
    else:
        print("  STATUS : ✗ SOME TESTS FAILED")
        print("  Check the mismatches above")

    print("=" * 60)

    return results, all_passed


# ─────────────────────────────────────────────────────────
# VALIDATION TESTS — Test error handling
# ─────────────────────────────────────────────────────────
def run_validation_tests(api_url="http://localhost:5000"):

    print("\n" + "=" * 60)
    print("  VALIDATION & ERROR HANDLING TESTS")
    print("=" * 60)

    validation_cases = [
        {
            "name"           : "Missing fields",
            "input"          : {"L": 2500, "R": 15},
            "expected_status": 400
        },
        {
            "name"           : "Wrong data type",
            "input"          : {"L": "hello", "R": 15, "F": 180, "M": 420000, "C": 0.95},
            "expected_status": 400
        },
        {
            "name"           : "Out of range value",
            "input"          : {"L": -500, "R": 15, "F": 180, "M": 420000, "C": 0.95},
            "expected_status": 400
        },
        {
            "name"           : "Valid request",
            "input"          : {"L": 2500, "R": 15, "F": 180, "M": 420000, "C": 0.95},
            "expected_status": 200
        },
    ]

    all_passed = True

    for case in validation_cases:
        try:
            response = requests.post(
                f"{api_url}/predict",
                json=case['input'],
                timeout=10
            )

            status   = response.status_code
            expected = case['expected_status']
            match    = (status == expected)

            if match:
                print(f"  ✓ {case['name']}")
                print(f"    Expected {expected} → Got {status} — PASS")
            else:
                print(f"  ✗ {case['name']}")
                print(f"    Expected {expected} → Got {status} — FAIL")
                all_passed = False

        except requests.exceptions.ConnectionError:
            print(f"  ✗ Cannot connect to API at {api_url}")
            all_passed = False

    print("=" * 60)
    return all_passed


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    API_URL = "http://localhost:5000"

    # Run prediction consistency tests
    results, pred_passed = run_e2e_tests(API_URL)

    # Run validation tests
    val_passed = run_validation_tests(API_URL)

    # Final verdict
    print("\n" + "=" * 60)
    print("  FINAL VERDICT")
    print("=" * 60)
    print(f"  Prediction consistency : {'✓ PASSED' if pred_passed else '✗ FAILED'}")
    print(f"  Validation handling    : {'✓ PASSED' if val_passed  else '✗ FAILED'}")

    if pred_passed and val_passed:
        print("\n  ✓ END-TO-END INTEGRATION COMPLETE")
        print("  Pipeline is consistent across all layers")
    else:
        print("\n  ✗ INTEGRATION ISSUES FOUND")
        print("  Check failures above")

    print("=" * 60)