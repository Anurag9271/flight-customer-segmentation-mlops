# main.py
# ─────────────────────────────────────────────────────────
# Entry point for the flight customer segmentation pipeline
# Run: python main.py
# Docker: docker run ... mlops-flight-segmentation
# ─────────────────────────────────────────────────────────

import os
import sys

# Make sure src/ is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline

if __name__ == "__main__":

    print("=" * 55)
    print("  Flight Customer Segmentation — Pipeline")
    print("=" * 55)

    INPUT_PATH  = "data/raw/flight_train.csv"
    OUTPUT_PATH = "outputs/clusters/dataset_with_clusters.csv"

    # Check data exists
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Data file not found at {INPUT_PATH}")
        print("Make sure you mounted the data folder correctly:")
        print("  docker run -v %cd%/data:/app/data ...")
        sys.exit(1)

    result = run_pipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        n_clusters=4
    )

    print("\nPipeline complete!")
    print(f"Customers segmented : {len(result):,}")
    print(f"Output saved to     : {OUTPUT_PATH}")