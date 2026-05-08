# monitoring/monitor_custom.py
# Custom drift detection using KS test — no EvidentlyAI needed
# Run: python monitoring/monitor_custom.py

import pandas as pd
import numpy as np
import joblib
import os
from scipy import stats

# ── Paths ─────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "outputs", "models")
DATA_PATH = os.path.join(BASE, "outputs", "clusters", "lrfmc_scaled.csv")

FEATURE_COLUMNS = ['L', 'R', 'F', 'M', 'C']

CLUSTER_NAMES = {
    0: "Loyal Regulars",
    1: "Occasional Leisure Flyers",
    2: "Champions",
    3: "At-Risk Customers"
}

# ── Load data ──────────────────────────────────────────────
print("Loading reference data...")
reference = pd.read_csv(DATA_PATH)[FEATURE_COLUMNS]

print("Simulating current data with drift...")
current        = reference.sample(5000, random_state=42).copy()
current['R']   = current['R'] * 1.3
current['F']   = current['F'] * 0.8
current['M']   = current['M'] * 1.2

# ── KS Test for each feature ───────────────────────────────
# KS Test = Kolmogorov-Smirnov test
# Compares two distributions statistically
# p-value < 0.05 = distributions are significantly different = DRIFT
print("\n" + "=" * 60)
print("DATA DRIFT REPORT — KS Test")
print("=" * 60)
print(f"{'Status':<15} {'Feature':<10} {'p-value':>10}")
print("-" * 40)

drifted = []
for col in FEATURE_COLUMNS:
    ks_stat, p_value = stats.ks_2samp(
        reference[col],
        current[col]
    )
    is_drifted = p_value < 0.05
    status     = "DRIFTED" if is_drifted else "OK"
    print(f"{status:<15} {col:<10} {p_value:>10.4f}")
    if is_drifted:
        drifted.append(col)

print("=" * 60)
print(f"Total drifted: {len(drifted)}/{len(FEATURE_COLUMNS)} features")
print(f"Drifted: {drifted}")

# ── Load models ────────────────────────────────────────────
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
km_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))

# ── Prediction distribution comparison ────────────────────
print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

for label, data in [("Reference", reference.sample(5000, random_state=42)),
                    ("Current",   current)]:
    scaled      = scaler.transform(data[FEATURE_COLUMNS])
    scaled_df   = pd.DataFrame(scaled, columns=FEATURE_COLUMNS)
    predictions = km_model.predict(scaled_df)

    print(f"\n{label} Data:")
    counts = pd.Series(predictions).value_counts().sort_index()
    for cluster_id, count in counts.items():
        pct  = count / len(predictions) * 100
        name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        print(f"  {name:<28}: {count:>5} ({pct:.1f}%)")

print("\nMonitoring complete!")