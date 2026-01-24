"""
Routine Feature Engineering Module
Project: Stress Management System using Face Recognition + Daily Routine

Responsibilities:
- Validate daily routine feature ranges
- Normalize / clamp values to realistic bounds
- Perform lightweight feature engineering (rule-based)

This module is used BEFORE model prediction.
"""

from typing import Dict
from src.config.config import BEHAVIORAL_FEATURES


# =============================
# Acceptable Ranges (Domain Knowledge)
# =============================

FEATURE_RANGES = {
    "sleep_hours": (0.0, 12.0),
    "sleep_quality": (1, 5),
    "sleep_consistency": (0, 1),
    "daily_study_hours": (0.0, 14.0),
    "assignment_pressure": (1, 5),
    "physical_activity_minutes": (0.0, 180.0),
    "screen_time_hours": (0.0, 16.0),
    "social_interaction_time": (0.0, 300.0),
    "caffeine_intake_mg": (0.0, 600.0)
}


# =============================
# Validation & Normalization
# =============================

def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)


def validate_and_normalize_routine_features(routine_data: Dict) -> Dict:
    """
    Ensures routine features:
    - Exist
    - Are within realistic bounds

    Parameters:
    routine_data (dict): Raw routine input

    Returns:
    dict: Cleaned and validated routine features
    """
    cleaned = {}

    for feature in BEHAVIORAL_FEATURES:
        if feature not in routine_data:
            raise ValueError(f"Missing routine feature: {feature}")

        value = routine_data[feature]
        min_val, max_val = FEATURE_RANGES[feature]

        cleaned[feature] = clamp(float(value), min_val, max_val)

    return cleaned


if __name__ == "__main__":
    sample = {
        "sleep_hours": 7.5,
        "sleep_quality": 4,
        "sleep_consistency": 1,
        "daily_study_hours": 5,
        "assignment_pressure": 3,
        "physical_activity_minutes": 30,
        "screen_time_hours": 7,
        "social_interaction_time": 45,
        "caffeine_intake_mg": 180
    }

    print(validate_and_normalize_routine_features(sample))
