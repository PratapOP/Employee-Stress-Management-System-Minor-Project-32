"""
Utility Helper Functions
Project: Stress Management System using Face Recognition + Daily Routine

Responsibilities:
- Safe dictionary operations
- Feature fusion (routine + facial + emotion)
- Common validation utilities

This module contains reusable helpers used across the project.
"""

from typing import Dict, List


def safe_merge_dicts(*dicts: Dict) -> Dict:
    """
    Safely merges multiple dictionaries.
    Later dictionaries override earlier ones if keys overlap.
    """
    merged = {}
    for d in dicts:
        if not isinstance(d, dict):
            raise TypeError("All inputs to safe_merge_dicts must be dictionaries")
        merged.update(d)
    return merged


def validate_required_features(feature_dict: Dict, required_features: List[str]) -> None:
    """
    Ensures all required features are present in a dictionary.
    Raises ValueError if any feature is missing.
    """
    missing = [f for f in required_features if f not in feature_dict]
    if missing:
        raise ValueError(f"Missing required features: {missing}")


def order_features(feature_dict: Dict, feature_order: List[str]) -> List[float]:
    """
    Orders features according to a predefined feature list.
    Returns a list suitable for ML model input.
    """
    validate_required_features(feature_dict, feature_order)
    return [float(feature_dict[f]) for f in feature_order]


def pretty_print_dict(data: Dict, indent: int = 0) -> None:
    """
    Nicely prints a dictionary (useful for CLI debugging).
    """
    for key, value in data.items():
        print(" " * indent + f"{key}: {value}")


if __name__ == "__main__":
    a = {"x": 1, "y": 2}
    b = {"y": 3, "z": 4}

    merged = safe_merge_dicts(a, b)
    pretty_print_dict(merged)
