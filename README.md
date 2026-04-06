# ✈️ Airline Customer Segmentation — Unsupervised ML Project

## Overview

This project applies unsupervised machine learning to segment 55,000 frequent flyer
members of a Chinese airline into meaningful behavioral groups using the **LRFMC framework**
(Length, Recency, Frequency, Monetary, discount sensitivity).

The goal is to discover hidden patterns in customer behavior — without any pre-existing
labels — and translate those patterns into actionable loyalty strategy recommendations.

---

## Business Problem

Airlines operate data-rich loyalty programs but often treat all members identically.
This wastes marketing spend and misses opportunities to retain high-value customers,
reactivate lapsed ones, and convert occasional flyers into loyal members.

**Core question:** Who are our customers, and what does each group actually need?

---

## Dataset

| Property | Value |
|---|---|
| Source | Airline Frequent Flyer Program (FFP) |
| Records | 55,000 customers |
| Features | 23 raw columns |
| Observation date | March 31, 2014 |
| Target variable | None (unsupervised) |

---

## LRFMC Framework

The five engineered features used for clustering:

| Dimension | Meaning | Source Feature(s) |
|---|---|---|
| **L** — Length | Membership duration (days) | `FFP_DATE`, `LOAD_TIME` |
| **R** — Recency | Days since last flight | `LAST_TO_END` |
| **F** — Frequency | Total flights taken | `FLIGHT_COUNT` |
| **M** — Monetary | Total kilometers flown | `SEG_KM_SUM` |
| **C** — Discount | Average discount coefficient | `avg_discount` |

---

## Algorithms Applied

| Algorithm | Purpose |
|---|---|
| **K-Means** | Primary clustering; optimal K via Elbow + Silhouette |
| **Agglomerative Hierarchical** | Dendrogram validation; Ward linkage |
| **DBSCAN** | Outlier/anomaly detection; density-based clusters |
| **PCA** | Dimensionality reduction for 2D visualization |

---

## Project Structure

```
flight_segmentation/
│
├── data/
│   ├── raw/                    ← Original flight_train.csv
│   └── processed/              ← Cleaned + LRFMC engineered data
│
├── notebooks/
│   └── project_analysis.ipynb  ← Full analysis report (9 sections)
│
├── src/
│   ├── preprocessing.py        ← Data cleaning functions
│   ├── feature_engineering.py  ← LRFMC builder + PCA
│   ├── clustering.py           ← K-Means, Agglomerative, DBSCAN
│   ├── evaluation.py           ← Metrics + visualization
│   └── pipeline.py             ← End-to-end pipeline
│
├── outputs/
│   ├── models/                 ← Saved model objects (.pkl)
│   ├── clusters/               ← dataset_with_clusters.csv
│   └── reports/                ← All saved plots/figures
│
├── requirements.txt
├── README.md
└── main.py                     ← Run full pipeline from CLI
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```

### 3. Explore the notebook
```bash
jupyter notebook notebooks/project_analysis.ipynb
```

---

## Deliverables

- `outputs/clusters/dataset_with_clusters.csv` — Final dataset with cluster labels
- `outputs/reports/` — All EDA and clustering visualizations
- `outputs/models/` — Saved KMeans and scaler objects
- `notebooks/project_analysis.ipynb` — Full documented analysis

---

## Expected Customer Segments

| Segment | Profile |
|---|---|
| VIP Champions | High frequency, high km, recently active, full fare |
| Loyal Regulars | Consistent rhythm, moderate-high engagement |
| Potential Loyalists | Newer members, growing activity |
| At-Risk / Lapsing | Historically active, not flown recently |
| Occasional Leisure | Low frequency, high discount sensitivity |
| Points Hoarders | High points, very low redemption activity |

---

## Author

Unsupervised Machine Learning Project — Customer Segmentation  
Dataset: Airline Frequent Flyer Program (55,000 members)
