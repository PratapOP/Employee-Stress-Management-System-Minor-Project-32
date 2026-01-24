"""
Step 5: Model Training
Project: Stress Management System using Face Recognition + Daily Routine

Responsibilities:
- Load dataset
- Preprocess data
- Train baseline ML model
- Save trained classifier
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.config.config import MODEL_TYPE, MODEL_OUTPUT_PATH
from src.data.load_data import load_and_validate_dataset, split_features_and_target
from src.data.preprocess import Preprocessor


def get_model():
    """
    Returns the ML model based on configuration.
    """
    if MODEL_TYPE == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    elif MODEL_TYPE == "svm":
        return SVC(kernel="rbf", probability=True, class_weight="balanced")
    elif MODEL_TYPE == "logistic":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")


def train():
    """
    Executes the full training pipeline.
    """
    # Load and validate dataset
    df = load_and_validate_dataset()

    # Split features and target
    X, y = split_features_and_target(df)

    # Preprocess
    preprocessor = Preprocessor()
    X = preprocessor.handle_missing_values(X)
    X_train, X_test, y_train, y_test = preprocessor.split_dataset(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

    # Train model
    model = get_model()
    model.fit(X_train_scaled, y_train)

    # Persist model and scaler
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    preprocessor.save_scaler()

    return model


if __name__ == "__main__":
    train()
