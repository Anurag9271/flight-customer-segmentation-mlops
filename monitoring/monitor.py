# monitoring/monitor.py
# Generates EvidentlyAI HTML drift report
# Run: python monitoring/monitor.py

import pandas as pd
import joblib
import os
import numpy as np

# ── Paths ─────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "outputs", "models")
DATA_PATH = os.path.join(BASE, "outputs", "clusters", "lrfmc_scaled.csv")

FEATURE_COLUMNS = ['L', 'R', 'F', 'M', 'C']

# ── Load data ──────────────────────────────────────────────
print("Loading reference data...")
reference_data = pd.read_csv(DATA_PATH)[FEATURE_COLUMNS]
print(f"Reference shape: {reference_data.shape}")

# ── Simulate current data with drift ──────────────────────
print("Simulating current data...")
current_data        = reference_data.sample(5000, random_state=42).copy()
current_data['R']   = current_data['R'] * 1.3   # recency drift
current_data['F']   = current_data['F'] * 0.8   # frequency drift
current_data['M']   = current_data['M'] * 1.2   # monetary drift

# ── Load models ────────────────────────────────────────────
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
km_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))

# ── Add predictions ────────────────────────────────────────
def get_predictions(df):
    scaled = scaler.transform(df[FEATURE_COLUMNS])
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS)
    return km_model.predict(scaled_df)

ref_sample = reference_data.sample(5000, random_state=42).copy()
ref_sample['Cluster'] = get_predictions(ref_sample)
current_data['Cluster'] = get_predictions(current_data)

# ── Generate Evidently report ──────────────────────────────
try:
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    print("Generating drift report...")
    report   = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    snapshot = report.run(
        reference_data=ref_sample,
        current_data=current_data
    )

    os.makedirs(os.path.join(BASE, "reports", "monitoring"), exist_ok=True)
    snapshot.save_html(
        os.path.join(BASE, "reports", "monitoring", "drift_report.html")
    )
    print("Report saved → reports/monitoring/drift_report.html")

except Exception as e:
    print(f"EvidentlyAI error: {e}")
    print("Run monitor_custom.py instead")