"""
Step 7: Stress Level Prediction (Inference)
Project: Stress Management System using Face Recognition + Daily Routine

Responsibilities:
- Load trained model and scaler
- Accept new feature input
- Predict stress level
"""

import joblib
import numpy as np

from src.config.config import MODEL_OUTPUT_PATH, BEHAVIORAL_FEATURES, FACIAL_FEATURES, EMOTION_FEATURES

SCALER_PATH = "models/scaler.pkl"

STRESS_MAPPING = {
    0: "Low Stress",
    1: "Moderate Stress",
    2: "High Stress"
}


def predict_stress(feature_dict: dict):
    """
    Predicts stress level for a single individual.

    Parameters:
    feature_dict (dict): Dictionary containing all required feature values

    Returns:
    str: Predicted stress level label
    """
    required_features = BEHAVIORAL_FEATURES + FACIAL_FEATURES + EMOTION_FEATURES

    # Ensure all features are present
    missing = [f for f in required_features if f not in feature_dict]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Arrange features in correct order
    feature_vector = np.array([[feature_dict[f] for f in required_features]])

    # Load scaler and model
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_OUTPUT_PATH)

    # Scale input
    feature_vector_scaled = scaler.transform(feature_vector)

    # Predict
    prediction = model.predict(feature_vector_scaled)[0]

    return STRESS_MAPPING.get(prediction, "Unknown")


if __name__ == "__main__":
    # Example usage
    sample_input = {
        "sleep_hours": 6.5,
        "sleep_quality": 3,
        "sleep_consistency": 1,
        "daily_study_hours": 5,
        "assignment_pressure": 4,
        "physical_activity_minutes": 20,
        "screen_time_hours": 7.5,
        "social_interaction_time": 40,
        "caffeine_intake_mg": 180,
        "eye_blink_rate": 22,
        "eye_opening_ratio": 0.28,
        "eyebrow_inner_distance": 32.1,
        "mouth_width_ratio": 0.62,
        "lip_compression_ratio": 0.19,
        "jaw_clench_ratio": 0.44,
        "face_symmetry_score": 0.93,
        "facial_movement_variance": 0.55,
        "expression_change_rate": 0.35,
        "emotion_angry": 0.15,
        "emotion_fear": 0.32,
        "emotion_sad": 0.22,
        "emotion_neutral": 0.20,
        "emotion_happy": 0.08,
        "emotion_surprise": 0.03
    }

    result = predict_stress(sample_input)
    print(f"Predicted Stress Level: {result}")
