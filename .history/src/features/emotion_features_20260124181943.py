"""
Emotion Feature Extraction Module
Project: Stress Management System using Face Recognition + Daily Routine

Responsibilities:
- Detect facial emotions from an image/frame
- Output normalized emotion probability scores
- Keep module independent from stress prediction logic

NOTE:
- Uses a lightweight pre-trained FER model
- Designed to be plug-and-play with facial_features.py
"""

import cv2
import numpy as np
from typing import Dict

try:
    from fer import FER
except ImportError:
    raise ImportError(
        "FER library not installed. Install using: pip install fer"
    )


# Initialize emotion detector
emotion_detector = FER(mtcnn=True)

# Expected emotion keys (must match config.EMOTION_FEATURES)
EMOTION_KEYS = [
    "emotion_angry",
    "emotion_fear",
    "emotion_sad",
    "emotion_neutral",
    "emotion_happy",
    "emotion_surprise"
]


# =============================
# Emotion Extraction Logic
# =============================

def extract_emotion_features(frame: np.ndarray) -> Dict:
    """
    Extracts emotion probability scores from a face image.

    Parameters:
    frame (np.ndarray): BGR image frame (OpenCV format)

    Returns:
    dict: Emotion probability features
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    emotions = emotion_detector.detect_emotions(rgb_frame)

    # Default neutral state (no face detected)
    emotion_features = {
        "emotion_angry": 0.0,
        "emotion_fear": 0.0,
        "emotion_sad": 0.0,
        "emotion_neutral": 1.0,
        "emotion_happy": 0.0,
        "emotion_surprise": 0.0
    }

    if not emotions:
        return emotion_features

    emotion_scores = emotions[0]["emotions"]

    # Map FER output to required feature names
    emotion_features = {
        "emotion_angry": round(emotion_scores.get("angry", 0.0), 4),
        "emotion_fear": round(emotion_scores.get("fear", 0.0), 4),
        "emotion_sad": round(emotion_scores.get("sad", 0.0), 4),
        "emotion_neutral": round(emotion_scores.get("neutral", 0.0), 4),
        "emotion_happy": round(emotion_scores.get("happy", 0.0), 4),
        "emotion_surprise": round(emotion_scores.get("surprise", 0.0), 4)
    }

    return emotion_features


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotions = extract_emotion_features(frame)
        cv2.putText(
            frame,
            str(emotions),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        cv2.imshow("Emotion Feature Extraction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
