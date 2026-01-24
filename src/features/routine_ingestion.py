"""
Daily Routine Data Collection & Ingestion
Project: Stress Management System using Face Recognition + Daily Routine

Responsibilities:
- Define daily routine schema
- Validate user-provided routine data
- Convert routine input into model-ready feature dictionary

NOTE:
- This module does NOT collect data directly (e.g., Google Form UI)
- It ingests already-collected routine data (JSON / dict / CSV row)
"""

from typing import Dict

# =============================
# Daily Routine Feature Schema
# =============================

ROUTINE_FEATURES = {
    "sleep_hours": float,
    "sleep_quality": int,
    "sleep_consistency": int,
    "daily_study_hours": float,
    "assignment_pressure": int,
    "physical_activity_minutes": float,
    "screen_time_hours": float,
    "social_interaction_time": float,
    "caffeine_intake_mg": float
}


# =============================
# Validation & Ingestion Logic
# =============================

def validate_routine_data(routine_data: Dict) -> None:
    """
    Validates routine data against expected schema.
    Raises ValueError if validation fails.
    """
    for feature, expected_type in ROUTINE_FEATURES.items():
        if feature not in routine_data:
            raise ValueError(f"Missing routine feature: {feature}")
        if not isinstance(routine_data[feature], expected_type):
            raise TypeError(
                f"Invalid type for {feature}: expected {expected_type.__name__}, "
                f"got {type(routine_data[feature]).__name__}"
            )


def ingest_routine_data(routine_data: Dict) -> Dict:
    """
    Validates and returns cleaned routine data.

    Parameters:
    routine_data (dict): Raw routine input (e.g., from form or API)

    Returns:
    dict: Cleaned routine features ready for model fusion
    """
    validate_routine_data(routine_data)

    cleaned_data = {feature: routine_data[feature] for feature in ROUTINE_FEATURES}

    return cleaned_data


if __name__ == "__main__":
    # Example usage
    sample_routine = {
        "sleep_hours": 6.5,
        "sleep_quality": 3,
        "sleep_consistency": 1,
        "daily_study_hours": 5.0,
        "assignment_pressure": 4,
        "physical_activity_minutes": 25.0,
        "screen_time_hours": 7.2,
        "social_interaction_time": 40.0,
        "caffeine_intake_mg": 180.0
    }

    routine_features = ingest_routine_data(sample_routine)
    print("Validated Routine Data:", routine_features)
