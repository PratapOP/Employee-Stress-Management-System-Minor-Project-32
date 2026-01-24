"""
Model Evaluation
Project: Stress Management System using Face Recognition + Daily Routine

Responsibilities:
- Load trained model and scaler
- Evaluate model performance on test data
- Generate standard ML metrics
"""

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config.config import MODEL_OUTPUT_PATH
from src.data.load_data import load_and_validate_dataset, split_features_and_target
from src.data.preprocess import Preprocessor


def evaluate():
    """
    Evaluates the trained model on the test dataset.
    """
    # Load dataset
    df = load_and_validate_dataset()
    X, y = split_features_and_target(df)

    # Preprocess
    preprocessor = Preprocessor()
    X = preprocessor.handle_missing_values(X)
    X_train, X_test, y_train, y_test = preprocessor.split_dataset(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

    # Load trained model
    model = joblib.load(MODEL_OUTPUT_PATH)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("Model Evaluation Results")
    print("------------------------")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)


if __name__ == "__main__":
    evaluate()
