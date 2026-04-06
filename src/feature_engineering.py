# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_membership_length(df):
    """
    L — Length: how many days since the customer joined the FFP.
    Computed from FFP_DATE to LOAD_TIME (the snapshot date).
    Longer = more established, loyal member.
    """
    df['FFP_DATE'] = pd.to_datetime(df['FFP_DATE'], format='mixed')
    df['LOAD_TIME'] = pd.to_datetime(df['LOAD_TIME'], format='mixed')
    df['L'] = (df['LOAD_TIME'] - df['FFP_DATE']).dt.days
    df = df.drop(columns=['FFP_DATE', 'LOAD_TIME', 'FIRST_FLIGHT_DATE'])
    return df


def compute_recency(df):
    """
    R — Recency: days since last flight.
    Already computed in raw data as LAST_TO_END.
    Lower R = flew recently = more active customer.
    """
    df = df.rename(columns={'LAST_TO_END': 'R'})
    return df


def compute_frequency(df):
    """
    F — Frequency: total number of flights taken.
    Directly from FLIGHT_COUNT.
    Higher F = frequent flyer.
    """
    df = df.rename(columns={'FLIGHT_COUNT': 'F'})
    return df


def compute_monetary(df):
    """
    M — Monetary: total kilometers flown.
    SEG_KM_SUM captures both volume and distance per trip.
    Richer signal than revenue alone.
    """
    df = df.rename(columns={'SEG_KM_SUM': 'M'})
    return df


def compute_discount(df):
    """
    C — Discount coefficient: avg ticket discount rate.
    Near 1.0 = full fare payer (premium customer).
    Below 0.7 = heavily discounted (price sensitive).
    From correlation analysis: this is independent of all
    other features — it adds unique information.
    """
    df = df.rename(columns={'avg_discount': 'C'})
    return df


def apply_log_transform(df):
    """
    Apply log1p transform to highly skewed LRFMC features.
    From EDA: F, M had skewness > 3.0 which distorts distances
    in K-Means. Log transform brings them closer to symmetric.
    R and L are moderately skewed — also transformed for consistency.
    C (avg_discount) skewness was 0.96 — nearly symmetric, skip it.
    """
    cols_to_transform = ['L', 'R', 'F', 'M']
    df[cols_to_transform] = np.log1p(df[cols_to_transform])
    return df


def drop_non_lrfmc_columns(df):
    """
    Keep only the 5 LRFMC features for clustering.
    Drop everything else — redundant from correlation analysis:
    BP_SUM and Points_Sum (r=0.92 with each other)
    SUM_YR_1, SUM_YR_2 (captured by M)
    All remaining raw columns not part of LRFMC.
    """
    lrfmc_cols = ['L', 'R', 'F', 'M', 'C']
    return df[lrfmc_cols]


def scale_features(df):
    """
    Apply StandardScaler to all LRFMC features.
    Without scaling, M (km in thousands) would dominate
    distance calculations over C (a decimal 0–1.5).
    Returns scaled DataFrame and the fitted scaler object.
    Scaler is saved so we can transform new customers later.
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )
    return X_scaled, scaler


def engineer_features(df):
    df = compute_membership_length(df)
    df = compute_recency(df)
    df = compute_frequency(df)
    df = compute_monetary(df)
    df = compute_discount(df)
    df = apply_log_transform(df)
    df = drop_non_lrfmc_columns(df)
    print(f"Feature engineering complete. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df