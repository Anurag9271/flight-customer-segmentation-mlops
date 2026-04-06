import pandas as pd
import numpy as np


def remove_duplicates(df):
    """Remove fully duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df)}")
    return df


def handle_missing_values(df):
    """
    Fill missing values with justified strategies.
    - SUM_YR_1, SUM_YR_2 : fill 0 (no activity that year)
    - AGE                 : fill median (robust to skew)
    - GENDER              : fill mode (only 3 nulls)
    """
    df['SUM_YR_1'] = df['SUM_YR_1'].fillna(0)
    df['SUM_YR_2'] = df['SUM_YR_2'].fillna(0)
    df['AGE'] = df['AGE'].fillna(df['AGE'].median())
    df['GENDER'] = df['GENDER'].fillna(df['GENDER'].mode()[0])
    return df


def remove_invalid_records(df):
    """
    Remove rows that are logically impossible.
    - Age below 18 or above 100 is implausible
    - Customers with zero km AND zero revenue have no behavioral data
    """
    before = len(df)
    df = df[(df['AGE'] >= 18) & (df['AGE'] <= 100)]
    df = df[~((df['SUM_YR_1'] == 0) & (df['SUM_YR_2'] == 0) & (df['SEG_KM_SUM'] == 0))]
    print(f"Invalid records removed: {before - len(df)}")
    return df


def drop_irrelevant_columns(df):
    """
    Drop columns that carry no signal for clustering.
    - MEMBER_NO         : just an ID
    - WORK_CITY         : high cardinality, too many nulls
    - WORK_PROVINCE     : high cardinality, too many nulls
    - WORK_COUNTRY      : almost all 'CN', no variance
    - LAST_FLIGHT_DATE  : already captured in LAST_TO_END
    """
    cols = [
        'MEMBER_NO', 'WORK_CITY', 'WORK_PROVINCE',
        'WORK_COUNTRY', 'LAST_FLIGHT_DATE'
    ]
    df = df.drop(columns=[c for c in cols if c in df.columns])
    return df


def preprocess(df):
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = remove_invalid_records(df)
    df = drop_irrelevant_columns(df)
    print(f"Preprocessing complete. Shape: {df.shape}")
    return df