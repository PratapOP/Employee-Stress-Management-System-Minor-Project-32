"""
Data Preprocessing
Project: Stress Management System using Face Recognition + Daily Routine

Responsibilities:
- Handle missing values
- Encode categorical features (if any)
- Scale numerical features
- Persist scaler for inference
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.config.config import (
    BEHAVIORAL_FEATURES,
    FACIAL_FEATURES,
    EMOTION_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
)


NUMERIC_FEATURES = BEHAVIORAL_FEATURES + FACIAL_FEATURES + EMOTION_FEATURES


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing numeric values using column-wise median.
        """
        df = df.copy()
        for col in NUMERIC_FEATURES:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        return df

    def split_dataset(self, X: pd.DataFrame, y: pd.Series):
        """
        Splits dataset into train and test sets.
        """
        return train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Fits scaler on training data and transforms both train and test sets.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def save_scaler(self, path: str = "models/scaler.pkl"):
        """
        Saves the fitted scaler for later inference use.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
