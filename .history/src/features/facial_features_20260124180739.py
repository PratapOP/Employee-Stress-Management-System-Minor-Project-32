"""
Step 11: Facial Feature Extraction Module
Project: Stress Management System using Face Recognition + Daily Routine

Responsibilities:
- Capture frames from webcam
- Detect face
- Extract facial landmarks
- Compute facial stress-related features

NOTE:
- This module focuses on feature extraction, not model prediction.
- Emotion probabilities can be integrated later or via a separate module.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Landmark indices (MediaPipe specific)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291]
JAW = [234, 454]


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_eye_opening_ratio(landmarks, eye_indices):
    vertical = euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    horizontal = euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return vertical / horizontal


def extract_facial_features(frame) -> Dict:
    """
    Extracts facial stress-related features from a single video frame.

    Returns:
    dict: Facial feature values
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = frame.shape

    landmarks = []
    for lm in face_landmarks.landmark:
        landmarks.append((int(lm.x * w), int(lm.y * h)))

    # Feature calculations
    left_eye_ratio = compute_eye_opening_ratio(landmarks, LEFT_EYE)
    right_eye_ratio = compute_eye_opening_ratio(landmarks, RIGHT_EYE)
    eye_opening_ratio = (left_eye_ratio + right_eye_ratio) / 2

    mouth_width = euclidean_distance(landmarks[MOUTH[0]], landmarks[MOUTH[1]])
    face_width = euclidean_distance(landmarks[JAW[0]], landmarks[JAW[1]])
    mouth_width_ratio = mouth_width / face_width

    features = {
        "eye_opening_ratio": round(eye_opening_ratio, 4),
        "mouth_width_ratio": round(mouth_width_ratio, 4)
    }

    return features


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_facial_features(frame)
        if features:
            cv2.putText(frame, str(features), (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Facial Feature Extraction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
