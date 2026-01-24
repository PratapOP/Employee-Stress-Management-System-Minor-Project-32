"""
Central Configuration File
Project: Stress Management System using Face Recognition + Daily Routine
"""

# ======================
# Paths
# ======================

DATASET_PATH = "data/sample/stress_multimodal_big_sample.csv"
MODEL_OUTPUT_PATH = "models/stress_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"

# ======================
# Training Parameters
# ======================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ======================
# Feature Groups
# ======================

# Daily Routine / Behavioral Features
BEHAVIORAL_FEATURES = [
    "sleep_hours",
    "sleep_quality",
    "sleep_consistency",
    "daily_study_hours",
    "assignment_pressure",
    "physical_activity_minutes",
    "screen_time_hours",
    "social_interaction_time",
    "caffeine_intake_mg"
]

# Facial Geometry Features
FACIAL_FEATURES = [
    "eye_opening_ratio",
    "mouth_width_ratio"
]

# Emotion Probability Features
EMOTION_FEATURES = [
    "emotion_angry",
    "emotion_fear",
    "emotion_sad",
    "emotion_neutral",
    "emotion_happy",
    "emotion_surprise"
]

# ======================
# Target Variable
# ======================

TARGET_COLUMN = "stress_level"

# ======================
# Model Selection
# ======================

MODEL_TYPE = "random_forest"  # options: random_forest | svm | logistic
