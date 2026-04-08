# src/log_experiment.py
# Run from project root: python src/log_experiment.py

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ── STEP 1: Point MLflow to a database file ───────────────
# This is the modern way — avoids all filesystem warnings
# mlflow.db will be created automatically in your project root
mlflow.set_tracking_uri("sqlite:///mlflow.db")
print("MLflow tracking URI set to: sqlite:///mlflow.db")

# ── STEP 2: Set the experiment name ───────────────────────
# All 6 runs will be grouped under this one experiment
mlflow.set_experiment("flight-customer-segmentation")
print("Experiment set: flight-customer-segmentation")

# ── STEP 3: Load your scaled data ─────────────────────────
print("\nLoading data...")
df = pd.read_csv("outputs/clusters/lrfmc_scaled.csv")
X  = df[['L', 'R', 'F', 'M', 'C']].values
print(f"Data loaded: {X.shape[0]} rows, {X.shape[1]} features")

# ── STEP 4: Define 6 experiments to run ───────────────────
runs = [
    {"n_clusters": 2, "init": "k-means++"},
    {"n_clusters": 3, "init": "k-means++"},
    {"n_clusters": 4, "init": "k-means++"},
    {"n_clusters": 4, "init": "random"},
    {"n_clusters": 5, "init": "k-means++"},
    {"n_clusters": 6, "init": "k-means++"},
]

print("\nStarting experiments...")
print("=" * 50)

# ── STEP 5: Loop and log each run ─────────────────────────
for r in runs:

    run_name = f"kmeans-k{r['n_clusters']}-{r['init']}"

    with mlflow.start_run(run_name=run_name):

        # Train the model
        model = KMeans(
            n_clusters=r["n_clusters"],
            init=r["init"],
            n_init=10,
            max_iter=300,
            random_state=42
        )
        labels = model.fit_predict(X)

        # Calculate metrics
        sil     = silhouette_score(X, labels,
                                   sample_size=10000,
                                   random_state=42)
        db      = davies_bouldin_score(X, labels)
        inertia = float(model.inertia_)

        # Log parameters (inputs)
        mlflow.log_param("n_clusters", r["n_clusters"])
        mlflow.log_param("init",       r["init"])
        mlflow.log_param("max_iter",   300)

        # Log metrics (outputs)
        mlflow.log_metric("silhouette_score",     round(sil, 4))
        mlflow.log_metric("davies_bouldin_index", round(db,  4))
        mlflow.log_metric("inertia",              round(inertia, 2))

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Print progress
        print(f"\n  {run_name}")
        print(f"  Silhouette  : {sil:.4f}")
        print(f"  DB Index    : {db:.4f}")
        print(f"  Inertia     : {inertia:,.0f}")

print("\n" + "=" * 50)
print("Done! All 6 runs logged.")
print("\nNow open a NEW terminal and run:")
print("mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("Then open: http://localhost:5000")