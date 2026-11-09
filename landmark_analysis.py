import numpy as np
from collections import deque

JOINTS = {
    "Left Elbow": (11, 13, 15),
    "Right Elbow": (12, 14, 16),
    "Left Shoulder": (13, 11, 23),
    "Right Shoulder": (14, 12, 24),
    "Left Hip": (11, 23, 25),
    "Right Hip": (12, 24, 26),
    "Left Knee": (23, 25, 27),
    "Right Knee": (24, 26, 28),
}
UPPER_BODY_JOINTS = ("Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder")
LOWER_BODY_JOINTS = ("Left Hip", "Right Hip", "Left Knee", "Right Knee")


def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees using 2D or 3D points."""
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    v1, v2 = a - b, c - b
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if mag1 < 1e-8 or mag2 < 1e-8:
        return None
    cosine = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cosine))), 2)


def _confidence_is_low(landmark, threshold):
    visibility = getattr(landmark, "visibility", None)
    if visibility is not None:
        try:
            return float(visibility) < threshold
        except (TypeError, ValueError):
            return True
    presence = getattr(landmark, "presence", None)
    if presence is not None:
        try:
            return float(presence) < threshold
        except (TypeError, ValueError):
            return True
    return False


def extract_joint_angles(landmarks, threshold=0.35, confidence_landmarks=None):
    """Coordinates from world landmarks, confidence from image landmarks."""
    empty = {name: None for name in JOINTS}
    if landmarks is None or len(landmarks) < 33:
        return empty
    if confidence_landmarks is not None and len(confidence_landmarks) < 33:
        confidence_landmarks = None
    angles = {}
    for name, indexes in JOINTS.items():
        coordinate_points = [landmarks[i] for i in indexes]
        confidence_points = (
            [confidence_landmarks[i] for i in indexes]
            if confidence_landmarks is not None else coordinate_points
        )
        if any(_confidence_is_low(p, threshold) for p in confidence_points):
            angles[name] = None
            continue
        try:
            coords = [[p.x, p.y, p.z] for p in coordinate_points]
            angles[name] = calculate_angle(*coords)
        except (AttributeError, TypeError, ValueError):
            angles[name] = None
    return angles


class AngleSmoother:
    """Smooth display values; keep current-frame validity separate."""
    def __init__(self, alpha=0.45, max_missing_frames=3):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be greater than 0 and at most 1")
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames cannot be negative")
        self.alpha = float(alpha)
        self.max_missing_frames = int(max_missing_frames)
        self.values = {name: None for name in JOINTS}
        self.missing_frames = {name: 0 for name in JOINTS}
        self.current_valid = {name: False for name in JOINTS}

    def reset(self):
        for name in JOINTS:
            self.values[name] = None
            self.missing_frames[name] = 0
            self.current_valid[name] = False

    def update(self, current_angles):
        stable = {}
        for name in JOINTS:
            current = current_angles.get(name)
            self.current_valid[name] = current is not None
            if current is not None:
                current = float(current)
                previous = self.values[name]
                value = current if previous is None else (
                    self.alpha * current + (1.0 - self.alpha) * previous
                )
                self.values[name] = value
                self.missing_frames[name] = 0
                stable[name] = round(value, 1)
            else:
                self.missing_frames[name] += 1
                if self.values[name] is not None and self.missing_frames[name] <= self.max_missing_frames:
                    stable[name] = round(self.values[name], 1)
                else:
                    self.values[name] = None
                    stable[name] = None
        return stable


# -------------------------
# BODY SEGMENT ANALYSIS
# -------------------------

BODY_SEGMENTS = {
    'left_upper_arm': (11, 13),
    'right_upper_arm': (12, 14),
    'left_forearm': (13, 15),
    'right_forearm': (14, 16),
    'left_thigh': (23, 25),
    'right_thigh': (24, 26),
    'left_shin': (25, 27),
    'right_shin': (26, 28),
    'torso_left': (11, 23),
    'torso_right': (12, 24),
}


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors. Returns float in [-1, 1]."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if mag1 < 1e-8 or mag2 < 1e-8:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0))


def compare_body_segments(current_landmarks, target_landmarks):
    """Compare body segment directions using cosine similarity.
    Returns dict mapping segment_name -> similarity score (0-100)."""
    if current_landmarks is None or target_landmarks is None:
        return {}
    scores = {}
    for name, (idx_a, idx_b) in BODY_SEGMENTS.items():
        try:
            c_a, c_b = current_landmarks[idx_a], current_landmarks[idx_b]
            t_a, t_b = target_landmarks[idx_a], target_landmarks[idx_b]
            current_vec = np.array([
                c_b.x - c_a.x, c_b.y - c_a.y,
                getattr(c_b, 'z', 0) - getattr(c_a, 'z', 0)
            ])
            target_vec = np.array([
                t_b.x - t_a.x, t_b.y - t_a.y,
                getattr(t_b, 'z', 0) - getattr(t_a, 'z', 0)
            ])
            sim = cosine_similarity(current_vec, target_vec)
            scores[name] = round(max(0.0, sim) * 100, 1)
        except (IndexError, AttributeError, TypeError):
            continue
    return scores


# -------------------------
# POSE STABILITY DETECTION
# -------------------------

class PoseStabilityDetector:
    """Detect when a pose is held stable for a duration."""

    def __init__(self, window_size=15, angle_threshold=5.0, required_stable_frames=10):
        self.window_size = window_size
        self.angle_threshold = angle_threshold
        self.required_stable_frames = required_stable_frames
        self.history = deque(maxlen=window_size)
        self.stable_count = 0

    def update(self, angles):
        """Update with current frame angles. Returns True if pose is stable."""
        self.history.append({k: v for k, v in angles.items()})
        if len(self.history) < 2:
            self.stable_count = 0
            return False
        prev = self.history[-2]
        curr = self.history[-1]
        max_change = 0.0
        compared = 0
        for name in JOINTS:
            p = prev.get(name)
            c = curr.get(name)
            if p is not None and c is not None:
                max_change = max(max_change, abs(float(c) - float(p)))
                compared += 1
        if compared < 3:
            self.stable_count = 0
            return False
        if max_change <= self.angle_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0
        return self.stable_count >= self.required_stable_frames

    def reset(self):
        """Clear history."""
        self.history.clear()
        self.stable_count = 0
