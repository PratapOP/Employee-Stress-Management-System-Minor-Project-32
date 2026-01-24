"""
Step 9: Stress Management Logic
Project: Stress Management System using Face Recognition + Daily Routine

Responsibilities:
- Translate predicted stress level into actionable outcomes
- Provide human-understandable recommendations
- Keep logic rule-based and explainable (non-ML)
"""

from datetime import datetime

# =============================
# Stress Threshold Definitions
# =============================

STRESS_ACTIONS = {
    "Low Stress": {
        "level": 0,
        "message": "You appear to be managing stress well.",
        "recommendations": [
            "Maintain your current routine",
            "Continue healthy sleep and activity habits"
        ],
        "alert": False
    },
    "Moderate Stress": {
        "level": 1,
        "message": "You are experiencing moderate stress.",
        "recommendations": [
            "Take short breaks between tasks",
            "Practice deep breathing for 5 minutes",
            "Reduce screen exposure temporarily"
        ],
        "alert": False
    },
    "High Stress": {
        "level": 2,
        "message": "High stress detected. Immediate attention recommended.",
        "recommendations": [
            "Stop current task and take a break",
            "Perform guided breathing or relaxation exercise",
            "Reach out to a trusted person if stress persists"
        ],
        "alert": True
    }
}


# =============================
# Core Stress Management Logic
# =============================

def generate_stress_response(stress_label: str):
    """
    Generates a stress management response based on predicted stress level.

    Parameters:
    stress_label (str): Output from ML model (Low/Moderate/High Stress)

    Returns:
    dict: Structured response for UI or logging
    """
    if stress_label not in STRESS_ACTIONS:
        raise ValueError(f"Unknown stress label: {stress_label}")

    action = STRESS_ACTIONS[stress_label]

    response = {
        "timestamp": datetime.now().isoformat(),
        "stress_level": stress_label,
        "message": action["message"],
        "recommendations": action["recommendations"],
        "alert_required": action["alert"]
    }

    return response


if __name__ == "__main__":
    # Example usage
    sample_prediction = "High Stress"
    result = generate_stress_response(sample_prediction)

    print("Stress Management Output")
    print("-------------------------")
    for key, value in result.items():
        print(f"{key}: {value}")
