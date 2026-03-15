import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Column names for CMAPSS dataset
def get_column_names():
    cols = ['unit_number', 'time_in_cycles']
    
    # operational settings
    for i in range(1, 4):
        cols.append(f'op_setting_{i}')
    
    # sensor measurements
    for i in range(1, 22):
        cols.append(f'sensor_{i}')
        
    return cols


# Load dataset
def load_data(train_path, test_path, rul_path):

    columns = get_column_names()

    print("\nLoading datasets...")
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, engine="python", encoding="latin1")
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, engine="python", encoding="latin1")

    print("Raw train shape:", train_df.shape)
    print("Raw test shape:", test_df.shape)

    # keep first 26 columns
    train_df = train_df.iloc[:, :26]
    test_df = test_df.iloc[:, :26]

    print("Trimmed train shape:", train_df.shape)
    print("Trimmed test shape:", test_df.shape)

    train_df.columns = columns
    test_df.columns = columns

    print("\nTrain columns assigned successfully")
    print(train_df.head())

    # Load test RUL
    rul_df = pd.read_csv(rul_path, header=None)
    rul_df.columns = ['RUL']

    print("\nRUL data preview:")
    print(rul_df.head())

    return train_df, test_df, rul_df


# Compute RUL for training dataset
def compute_rul(train_df):

    print("\nComputing RUL...")

    max_cycle = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycle.columns = ['unit_number', 'max_cycle']

    train_df = train_df.merge(max_cycle, on='unit_number')

    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']

    train_df.drop('max_cycle', axis=1, inplace=True)

    print("RUL computation completed")
    print(train_df[['unit_number', 'time_in_cycles', 'RUL']].head())

    return train_df


# Normalize sensor data
def normalize_data(train_df, test_df):

    print("\nNormalizing features...")

    scaler = MinMaxScaler()

    feature_cols = train_df.columns.difference(['unit_number', 'time_in_cycles', 'RUL'])

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    print("Normalization done")
    print("\nNormalized train sample:")
    print(train_df.head())

    return train_df, test_df, scaler