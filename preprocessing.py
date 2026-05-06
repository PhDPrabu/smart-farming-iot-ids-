"""
Preprocessing Pipeline for Smart Farming IoT Intrusion Detection

This module implements the preprocessing workflow described in:
Section 4.1 - Experimental Setup

Processing Steps:
1. Removal of inconsistent and missing records
2. Categorical feature encoding
3. Normalization of numerical attributes
4. Binary label conversion (Benign vs Attack)
5. Stratified train/validation/test partitioning
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_STATE = 42


def remove_inconsistent_records(df):
    """
    Remove inconsistent or missing records.
    """
    df = df.dropna()
    df = df.drop_duplicates()

    return df


def encode_categorical_features(df):
    """
    Perform categorical feature encoding.
    """

    categorical_columns = df.select_dtypes(include=["object"]).columns

    label_encoders = {}

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column].astype(str))

        label_encoders[column] = encoder

    return df, label_encoders


def normalize_numerical_features(df):
    """
    Normalize numerical attributes using StandardScaler.
    """

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    scaler = StandardScaler()

    df[numerical_columns] = scaler.fit_transform(
        df[numerical_columns]
    )

    return df, scaler


def convert_to_binary_labels(df, label_column="Label"):
    """
    Convert multi-class attack labels into binary labels:
    Benign = 0
    Attack = 1
    """

    df[label_column] = df[label_column].apply(
        lambda x: 0 if str(x).lower() == "benign" else 1
    )

    return df


def stratified_partition(X, y):
    """
    Perform stratified train/validation/test split
    using 70:10:20 ratio.
    """

    # Train split (70%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Validation/Test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.67,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def preprocessing_pipeline(df, label_column="Label"):
    """
    Complete preprocessing workflow.
    """

    print("Starting preprocessing pipeline...")

    df = remove_inconsistent_records(df)

    df = convert_to_binary_labels(df, label_column)

    df, encoders = encode_categorical_features(df)

    df, scaler = normalize_numerical_features(df)

    X = df.drop(columns=[label_column])

    y = df[label_column]

    split_data = stratified_partition(X, y)

    print("Preprocessing completed successfully.")

    return split_data, encoders, scaler


if __name__ == "__main__":

    print(
        "Smart Farming IoT IDS Preprocessing Module Initialized."
    )
