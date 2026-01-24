"""
Step 2: Dataset Loading & Validation
Project: Stress Management System using Face Recognition + Daily Routine

Responsibilities:
- Load CSV dataset
- Validate required columns
- Remove rows without consent
- Separate features (X) and target (y)
"""

import pandas as pd
from src.config.config import (
    DATASET_PATH,
    BEHAVIORAL_FEATURES,
    FACIAL_FEATURES,
    EMOTION_FEATURES,
    TARGET_COLUMN,
)


def load_and_validate_dataset(path: str = DATASET_PATH):
    """
    Loads the dataset and performs basic validation.
    Returns cleaned DataFrame.
    """
    df = pd.read_csv(path)

    # Required columns
    required_columns = (
        BEHAVIORAL_FEATURES
        + FACIAL_FEATURES
        + EMOTION_FEATURES
        + [TARGET_COLUMN, "consent_given"]
    )

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Keep only consented participants
    df = df[df["consent_given"] == 1].reset_index(drop=True)

    return df


def split_features_and_target(df: pd.DataFrame):
    """
    Splits dataframe into features (X) and target (y).
    """
    feature_columns = BEHAVIORAL_FEATURES + FACIAL_FEATURES + EMOTION_FEATURES

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    return X, y
