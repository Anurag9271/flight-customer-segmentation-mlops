import pandas as pd
import numpy as np
import joblib
import os

from src.preprocessing import preprocess
from src.feature_engineering import engineer_features, scale_features
from src.clustering import apply_pca, run_kmeans, run_hierarchical, run_dbscan
from src.evaluation import compare_all_algorithms


def run_pipeline(input_path, output_path, n_clusters=4):
    """
    End-to-end pipeline: raw CSV → clustered output CSV.
    Saves scaler, pca, and kmeans model to outputs/models/.
    All intermediate steps print progress so you can see what's happening.
    """

    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    os.makedirs('outputs/clusters', exist_ok=True)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(input_path)
    print(f"  Shape: {df.shape}")

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    print("\nPreprocessing...")
    df_clean = preprocess(df.copy())

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    print("\nEngineering LRFMC features...")
    df_lrfmc = engineer_features(df_clean.copy())

    # ── Step 4: Scale ─────────────────────────────────────────────────────────
    print("\nScaling features...")
    X_scaled, scaler = scale_features(df_lrfmc)

    # ── Step 5: PCA (for visualisation) ───────────────────────────────────────
    print("\nApplying PCA for visualisation...")
    X_pca, pca = apply_pca(X_scaled.values)

    # ── Step 6: Clustering ────────────────────────────────────────────────────
    print(f"\nClustering with K={n_clusters}...")

    km_model, km_labels       = run_kmeans(X_scaled.values, n_clusters=n_clusters)
    agg_model, agg_labels, agg_idx     = run_hierarchical(X_scaled.values, n_clusters=n_clusters)
    db_model, db_labels       = run_dbscan(X_scaled.values, eps=0.8, min_samples=10)

    # ── Step 7: Evaluate ──────────────────────────────────────────────────────
    print("\nEvaluating...")
    labels_dict = {
        'K-Means'      : km_labels,
        'Hierarchical' : agg_labels,
        'DBSCAN'       : db_labels
    }
    compare_all_algorithms(X_scaled.values, labels_dict)

    # ── Step 8: Save models ───────────────────────────────────────────────────
    print("\nSaving models...")
    joblib.dump(scaler,   'outputs/models/scaler.pkl')
    joblib.dump(pca,      'outputs/models/pca.pkl')
    joblib.dump(km_model, 'outputs/models/kmeans_model.pkl')
    print("  scaler.pkl, pca.pkl, kmeans_model.pkl saved.")

    # ── Step 9: Save final output ─────────────────────────────────────────────
    print("\nSaving results...")
    df_lrfmc['KMeans_Cluster']      = km_labels
    df_lrfmc['Hierarchical_Cluster'] = agg_labels
    df_lrfmc['DBSCAN_Cluster']      = db_labels
    df_lrfmc.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    print("\nPipeline complete!")
    return df_lrfmc