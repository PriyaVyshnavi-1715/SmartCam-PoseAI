"""
Sitting-posture measurements using visible MediaPipe Pose landmarks.

This module intentionally measures only what the camera can currently support:
- shoulder level (left/right shoulder line)
- head position relative to the shoulder midpoint

It does NOT claim to measure spinal curvature or full torso lean.
Thresholds are initial engineering heuristics and should be calibrated.
"""

import math


MIN_VISIBILITY = 0.35

# MediaPipe Pose landmark indexes
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def _visibility(point):
    """Read visibility only; presence can be low even when visibility is strong."""
    value = getattr(point, "visibility", None)
    if value is None:
        value = getattr(point, "presence", None)
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _visible(landmarks, index, threshold=MIN_VISIBILITY):
    return (
        landmarks is not None
        and len(landmarks) > index
        and _visibility(landmarks[index]) >= threshold
    )


def analyze_sitting_posture(landmarks, threshold=MIN_VISIBILITY):
    """
    Return measurable sitting-posture indicators from normalized image landmarks.

    shoulder_tilt_degrees:
        Signed angle of the shoulder line. Positive means the right shoulder
        appears lower in the image; negative means the left appears lower.

    head_offset:
        Nose horizontal offset from shoulder midpoint, normalized by shoulder
        width. Positive means the nose is to the right of the shoulder midpoint.

    These are 2D camera-view measurements, not clinical posture assessments.
    """
    result = {
        "ready": False,
        "shoulder_tilt_degrees": None,
        "head_offset": None,
        "feedback": ["Keep both shoulders visible to analyze sitting posture."],
        "measurement_note": (
            "2D camera estimates only; spinal curvature and full torso lean "
            "are not measured."
        ),
    }

    if not _visible(landmarks, LEFT_SHOULDER, threshold) or not _visible(
        landmarks, RIGHT_SHOULDER, threshold
    ):
        return result

    left = landmarks[LEFT_SHOULDER]
    right = landmarks[RIGHT_SHOULDER]

    dx = float(right.x) - float(left.x)
    dy = float(right.y) - float(left.y)
    shoulder_width = math.hypot(dx, dy)

    if shoulder_width < 1e-6:
        result["feedback"] = ["Move slightly farther from the camera."]
        return result

    tilt = math.degrees(math.atan2(dy, dx))
    tilt = (tilt + 90) % 180 - 90
    result["shoulder_tilt_degrees"] = round(tilt, 1)

    feedback = []
    if abs(tilt) <= 5.0:
        feedback.append("Shoulders appear approximately level.")
    elif tilt > 0:
        feedback.append("Your right shoulder appears lower; gently level your shoulders.")
    else:
        feedback.append("Your left shoulder appears lower; gently level your shoulders.")

    if _visible(landmarks, NOSE, threshold):
        nose = landmarks[NOSE]
        midpoint_x = (float(left.x) + float(right.x)) / 2.0
        offset = (float(nose.x) - midpoint_x) / shoulder_width
        result["head_offset"] = round(offset, 3)

        if abs(offset) <= 0.12:
            feedback.append("Your head appears centered between your shoulders.")
        elif offset > 0:
            feedback.append("Your head appears shifted to the right of your shoulder center.")
        else:
            feedback.append("Your head appears shifted to the left of your shoulder center.")
    else:
        feedback.append("Keep your head visible for head-alignment analysis.")

    result["ready"] = True
    result["feedback"] = feedback
    return result
