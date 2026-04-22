# api.py
# ─────────────────────────────────────────────────────────
# Flight Customer Segmentation — Prediction API
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
# We load ONCE here — not inside the predict function
# Reason: loading a pkl file takes time
# If we load on every request → very slow API
# Loading once at startup → instant predictions

print("Loading models...")

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "outputs", "models")

try:
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    km_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    print("scaler.pkl      loaded OK")
    print("kmeans_model.pkl loaded OK")
    MODELS_LOADED = True

except FileNotFoundError as e:
    print(f"ERROR loading models: {e}")
    print("Run main.py first to generate model files")
    scaler        = None
    km_model      = None
    MODELS_LOADED = False

# ── Segment names from your project ──────────────────────
# These are the 4 segments you found during clustering
# Cluster 0, 1, 2, 3 → meaningful names
SEGMENT_MAP = {
    0: "Loyal Regulars",
    1: "Occasional Leisure Flyers",
    2: "Champions",
    3: "At-Risk Customers"
}

# Segment business actions
ACTION_MAP = {
    0: "Push to next loyalty tier with milestone rewards",
    1: "Target with seasonal promotions and flash sales",
    2: "Retain with priority boarding and lounge access",
    3: "Launch win-back campaign with reactivation offers"
}

# All 5 fields the API expects
REQUIRED_FIELDS = ['L', 'R', 'F', 'M', 'C']

# ─────────────────────────────────────────────────────────
# ENDPOINT 1 — Health Check
# GET http://localhost:5000/health
#
# Purpose: confirm the API is running and models are loaded
# Used by: Docker health checks, CI pipelines, monitoring
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
        "dataset"      : "55000 airline customers"
    }), 200


# ─────────────────────────────────────────────────────────
# ENDPOINT 2 — Predict Customer Segment
# POST http://localhost:5000/predict
#
# Request body (JSON):
# {
#     "L": 2500,     days since FFP enrollment
#     "R": 15,       days since last flight
#     "F": 180,      total flights taken
#     "M": 420000,   total km flown
#     "C": 0.95      average discount coefficient
# }
#
# Response:
# {
#     "segment"   : "Champions",
#     "cluster_id": 2,
#     "action"    : "Retain with priority boarding..."
# }
# ─────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():

    # ── Guard: check models loaded ────────────────────────
    if not MODELS_LOADED:
        return jsonify({
            "error"  : "Models not loaded",
            "message": "Run main.py first to train and save models"
        }), 500

    # ── Step 1: Get JSON from request ─────────────────────
    data = request.get_json()

    # Check if any JSON was sent at all
    if not data:
        return jsonify({
            "error"  : "No JSON received",
            "message": "Send request with Content-Type: application/json"
        }), 400

    # ── Step 2: Check all required fields present ─────────
    missing_fields = [f for f in REQUIRED_FIELDS if f not in data]

    if missing_fields:
        return jsonify({
            "error"   : "Missing fields",
            "missing" : missing_fields,
            "required": REQUIRED_FIELDS,
            "message" : f"Provide all 5 LRFMC features"
        }), 400

    # ── Step 3: Check all values are numbers ─────────────
    for field in REQUIRED_FIELDS:
        value = data[field]
        if not isinstance(value, (int, float)):
            return jsonify({
                "error"  : f"Invalid value for '{field}'",
                "message": f"'{field}' must be a number, got: {type(value).__name__}"
            }), 400

    # ── Step 4: Extract raw LRFMC values ─────────────────
    L = float(data['L'])   # membership length in days
    R = float(data['R'])   # recency in days
    F = float(data['F'])   # flight count
    M = float(data['M'])   # total km
    C = float(data['C'])   # discount coefficient

    # ── Step 5: Apply log transform ───────────────────────
    # CRITICAL: must match exactly what was done in
    # feature_engineering.py during training
    # L, R, F, M were log transformed
    # C was NOT log transformed
    L_t = np.log1p(L)
    R_t = np.log1p(R)
    F_t = np.log1p(F)
    M_t = np.log1p(M)
    C_t = C

    # ── Step 6: Build DataFrame with column names ─────────
    # Must use column names because scaler was fitted
    # on a DataFrame with these exact column names
    features = pd.DataFrame(
        [[L_t, R_t, F_t, M_t, C_t]],
        columns=['L', 'R', 'F', 'M', 'C']
    )

    # ── Step 7: Scale using saved scaler ─────────────────
    # scaler knows the mean and std from training data
    # it applies the same transformation to new input
    features_scaled = scaler.transform(features)

    # ── Step 8: Predict cluster ───────────────────────────
    cluster_id = int(km_model.predict(features_scaled)[0])

    # ── Step 9: Map to segment name ───────────────────────
    segment = SEGMENT_MAP.get(cluster_id, "Unknown")
    action  = ACTION_MAP.get(cluster_id,  "No action defined")

    # ── Step 10: Return prediction ────────────────────────
    return jsonify({
        "status"    : "success",
        "cluster_id": cluster_id,
        "segment"   : segment,
        "action"    : action,
        "input_received": {
            "L": L, "R": R,
            "F": F, "M": M, "C": C
        }
    }), 200


# ─────────────────────────────────────────────────────────
# ENDPOINT 3 — Get All Segments Info
# GET http://localhost:5000/segments
#
# Purpose: show what all 4 segments mean
# ─────────────────────────────────────────────────────────
@app.route('/segments', methods=['GET'])
def segments():
    return jsonify({
        "total_segments": 4,
        "framework"     : "LRFMC",
        "segments": [
            {
                "cluster_id" : 0,
                "name"       : "Loyal Regulars",
                "share"      : "23.3%",
                "profile"    : "Active regular flyers, moderate-high engagement",
                "action"     : "Push to next loyalty tier"
            },
            {
                "cluster_id" : 1,
                "name"       : "Occasional Leisure Flyers",
                "share"      : "24.7%",
                "profile"    : "Price sensitive, low frequency, new members",
                "action"     : "Seasonal promotions and flash sales"
            },
            {
                "cluster_id" : 2,
                "name"       : "Champions",
                "share"      : "22.6%",
                "profile"    : "High frequency, high km, recently active, full fare",
                "action"     : "Retain — priority boarding and lounge access"
            },
            {
                "cluster_id" : 3,
                "name"       : "At-Risk Customers",
                "share"      : "29.5%",
                "profile"    : "Long-term members who have stopped flying recently",
                "action"     : "Win-back campaign with reactivation offers"
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