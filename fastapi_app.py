# fastapi.py

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Flight Customer Segmentation API",
    description="Predicts flight customer segment using KMeans clustering on LRFMC features",
    version="1.0.0"
)

# ── Load models ───────────────────────────────────────────
scaler   = joblib.load("outputs/models/scaler.pkl")
km_model = joblib.load("outputs/models/kmeans_model.pkl")

print(f"[INFO] Loaded model: {type(km_model).__name__}")

# ── Cluster information ───────────────────────────────────
CLUSTER_INFO = {
    0: "Loyal Regulars",
    1: "Occasional Leisure Flyers",
    2: "Champions",
    3: "At-Risk Customers"
}

CLUSTER_ACTION = {
    0: "Push to next loyalty tier with milestone rewards",
    1: "Target with seasonal promotions and flash sales",
    2: "Retain with priority boarding and lounge access",
    3: "Launch win-back campaign with reactivation offers"
}

# ── Pydantic Input Model ──────────────────────────────────
class CustomerInput(BaseModel):
    L: float = Field(..., ge=0,   le=10000,   description="Membership length in days")
    R: float = Field(..., ge=0,   le=1000,    description="Days since last flight")
    F: float = Field(..., ge=0,   le=500,     description="Total number of flights")
    M: float = Field(..., ge=0,   le=1000000, description="Total kilometers flown")
    C: float = Field(..., ge=0.0, le=2.0,     description="Average discount coefficient")

# ── Routes ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model" : "KMeans Flight Customer Segmentation"
    }

@app.post("/predict")
def predict(customer: CustomerInput):

    # ── Apply log transform (same as training) ────────────
    L_t = np.log1p(customer.L)
    R_t = np.log1p(customer.R)
    F_t = np.log1p(customer.F)
    M_t = np.log1p(customer.M)
    C_t = customer.C    # C was not log transformed in training

    # ── Build DataFrame ───────────────────────────────────
    input_data = pd.DataFrame(
        [[L_t, R_t, F_t, M_t, C_t]],
        columns=['L', 'R', 'F', 'M', 'C']
    )

    # ── Pipeline: scale → predict ─────────────────────────
    input_scaled = scaler.transform(input_data)
    cluster      = int(km_model.predict(input_scaled)[0])

    return {
        "cluster_id"  : cluster,
        "segment_name": CLUSTER_INFO[cluster],
        "action"      : CLUSTER_ACTION[cluster],
        "input_received": {
            "L": customer.L,
            "R": customer.R,
            "F": customer.F,
            "M": customer.M,
            "C": customer.C
        }
    }

@app.get("/segments")
def segments():
    return{
        "total segments: 4,"
        "framework":"LRFMC",
        "segments":[
            {
                "cluster_id":0,
                "name":"Loyal Regulars"
            },
            {
                "cluster_id": 1,
                "name":"Occasional Leisure Flyers"
            },
            {
                "cluster_id": 2,
                "name": "Champions"
            },
            {
            "cluster_id": 3,
            "name": "At-Risk Customers"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("fastapi:app", host="0.0.0.0", port=port, reload=False)