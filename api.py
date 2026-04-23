# api.py
# ─────────────────────────────────────────────────────────
# Flight Customer Segmentation — Prediction API
# Day 6 + Day 7: Full validation and error handling added
#
# Run : python api.py
# Test: POST http://localhost:5000/predict
# ─────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# ── Create Flask app ──────────────────────────────────────
app = Flask(__name__)

# ── Load models when API starts ───────────────────────────
print("Loading models...")

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "outputs", "models")

try:
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    km_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    print("scaler.pkl       loaded OK")
    print("kmeans_model.pkl loaded OK")
    MODELS_LOADED = True

except FileNotFoundError as e:
    print(f"ERROR loading models: {e}")
    print("Run main.py first to generate model files")
    scaler        = None
    km_model      = None
    MODELS_LOADED = False

# ── Segment names ─────────────────────────────────────────
SEGMENT_MAP = {
    0: "Loyal Regulars",
    1: "Occasional Leisure Flyers",
    2: "Champions",
    3: "At-Risk Customers"
}

ACTION_MAP = {
    0: "Push to next loyalty tier with milestone rewards",
    1: "Target with seasonal promotions and flash sales",
    2: "Retain with priority boarding and lounge access",
    3: "Launch win-back campaign with reactivation offers"
}

# ── Required fields ───────────────────────────────────────
REQUIRED_FIELDS = ['L', 'R', 'F', 'M', 'C']

# ── NEW: Acceptable value ranges for each feature ─────────
# Based on your actual dataset distribution
# Values outside these ranges are likely errors
RANGES = {
    'L': (0,    10000),   # membership days: 0 to ~27 years
    'R': (0,    1000),    # recency days: 0 to ~3 years
    'F': (0,    500),     # total flights: 0 to 500
    'M': (0,    1000000), # total km: 0 to 1 million
    'C': (0.0,  2.0)      # discount coefficient: 0 to 2
}

# ─────────────────────────────────────────────────────────
# ENDPOINT 1 — Health Check
# GET http://localhost:5000/health
# ─────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status"       : "healthy",
        "models_loaded": MODELS_LOADED,
        "model_type"   : "KMeans",
        "n_clusters"   : 4,
        "framework"    : "LRFMC",
        "features"     : REQUIRED_FIELDS,
        "dataset"      : "55000 airline customers",
        "validation"   : "presence + type + range checks enabled"
    }), 200


# ─────────────────────────────────────────────────────────
# ENDPOINT 2 — Predict Customer Segment
# POST http://localhost:5000/predict
# ─────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():

    # ── Guard: check models loaded ────────────────────────
    if not MODELS_LOADED:
        return jsonify({
            "error"  : "Models not loaded",
            "message": "Run main.py first to train and save models"
        }), 500

    # ─────────────────────────────────────────────────────
    # VALIDATION LAYER 1 — Check JSON exists
    # If no JSON body sent at all
    # ─────────────────────────────────────────────────────
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error"  : "No JSON received",
            "message": "Set Content-Type: application/json and send a JSON body"
        }), 400

    # ─────────────────────────────────────────────────────
    # VALIDATION LAYER 2 — Presence Check
    # Check all 5 required fields are present
    # ─────────────────────────────────────────────────────
    missing_fields = [f for f in REQUIRED_FIELDS if f not in data]

    if missing_fields:
        return jsonify({
            "error"   : "Missing required fields",
            "missing" : missing_fields,
            "required": REQUIRED_FIELDS,
            "message" : "All 5 LRFMC features must be provided"
        }), 400

    # ─────────────────────────────────────────────────────
    # VALIDATION LAYER 3 — Type Check
    # Check all values are numbers (int or float)
    # ─────────────────────────────────────────────────────
    for field in REQUIRED_FIELDS:
        value = data[field]
        if not isinstance(value, (int, float)):
            return jsonify({
                "error"   : f"Invalid type for '{field}'",
                "received": type(value).__name__,
                "expected": "number (int or float)",
                "message" : f"'{field}' must be a number, got '{type(value).__name__}'"
            }), 400

    # ─────────────────────────────────────────────────────
    # Extract values after validation passes
    # ─────────────────────────────────────────────────────
    L = float(data['L'])
    R = float(data['R'])
    F = float(data['F'])
    M = float(data['M'])
    C = float(data['C'])

    # ─────────────────────────────────────────────────────
    # VALIDATION LAYER 4 — Range Check
    # Check each value falls within acceptable bounds
    # ─────────────────────────────────────────────────────
    values = {'L': L, 'R': R, 'F': F, 'M': M, 'C': C}

    for field, (min_val, max_val) in RANGES.items():
        val = values[field]
        if not (min_val <= val <= max_val):
            return jsonify({
                "error"   : f"Value out of range for '{field}'",
                "received": val,
                "expected": f"Between {min_val} and {max_val}",
                "message" : f"'{field}' = {val} is outside the acceptable range"
            }), 400

    # ─────────────────────────────────────────────────────
    # PREDICTION — Wrapped in try-except
    # Even after validation passes, unexpected errors
    # can happen. We catch them and return clean messages.
    # ─────────────────────────────────────────────────────
    try:

        # Step 1: Log transform — same as training
        # L, R, F, M were log transformed in feature_engineering.py
        # C was NOT log transformed
        L_t = np.log1p(L)
        R_t = np.log1p(R)
        F_t = np.log1p(F)
        M_t = np.log1p(M)
        C_t = C

        # Step 2: Build DataFrame with correct column names
        features = pd.DataFrame(
            [[L_t, R_t, F_t, M_t, C_t]],
            columns=['L', 'R', 'F', 'M', 'C']
        )

        # Step 3: Scale using saved scaler
        features_scaled = scaler.transform(features)

        # Step 4: Predict cluster
        cluster_id = int(km_model.predict(features_scaled)[0])

        # Step 5: Map to segment name and action
        segment = SEGMENT_MAP.get(cluster_id, "Unknown")
        action  = ACTION_MAP.get(cluster_id,  "No action defined")

        # Step 6: Return successful prediction
        return jsonify({
            "status"        : "success",
            "cluster_id"    : cluster_id,
            "segment"       : segment,
            "action"        : action,
            "input_received": {
                "L": L, "R": R,
                "F": F, "M": M, "C": C
            }
        }), 200

    except Exception as e:
        # Log the real error privately for debugging
        print(f"[ERROR] Prediction failed: {e}")

        # Return clean message to user — no internal details
        return jsonify({
            "error"  : "Prediction failed",
            "message": "Internal server error. Check server logs."
        }), 500


# ─────────────────────────────────────────────────────────
# ENDPOINT 3 — Get All Segments Info
# GET http://localhost:5000/segments
# ─────────────────────────────────────────────────────────
@app.route('/segments', methods=['GET'])
def segments():
    return jsonify({
        "total_segments": 4,
        "framework"     : "LRFMC",
        "segments": [
            {
                "cluster_id": 0,
                "name"      : "Loyal Regulars",
                "share"     : "23.3%",
                "profile"   : "Active regular flyers, moderate-high engagement",
                "action"    : "Push to next loyalty tier"
            },
            {
                "cluster_id": 1,
                "name"      : "Occasional Leisure Flyers",
                "share"     : "24.7%",
                "profile"   : "Price sensitive, low frequency, new members",
                "action"    : "Seasonal promotions and flash sales"
            },
            {
                "cluster_id": 2,
                "name"      : "Champions",
                "share"     : "22.6%",
                "profile"   : "High frequency, high km, recently active, full fare",
                "action"    : "Retain — priority boarding and lounge access"
            },
            {
                "cluster_id": 3,
                "name"      : "At-Risk Customers",
                "share"     : "29.5%",
                "profile"   : "Long members who have stopped flying recently",
                "action"    : "Win-back campaign with reactivation offers"
            }
        ]
    }), 200


# ─────────────────────────────────────────────────────────
# Start the server
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  Flight Customer Segmentation API")
    print("=" * 55)
    print("  Health  : GET  http://localhost:5000/health")
    print("  Predict : POST http://localhost:5000/predict")
    print("  Segments: GET  http://localhost:5000/segments")
    print("=" * 55 + "\n")

    app.run(
        host ='0.0.0.0',
        port =5000,
        debug=True
    )