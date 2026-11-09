import cv2
import mediapipe as mp
import numpy as np
import ctypes
import time
from pathlib import Path

from landmark_analysis import extract_joint_angles, AngleSmoother, JOINTS
from pose_matching import REFERENCE_POSES, compare_pose
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. MODEL PATHS AND SETTINGS
# ==========================================

POSE_MODEL = Path("assets/pose_landmarker_lite.task")
HAND_MODEL = Path("assets/hand_landmarker.task")
FACE_MODEL = Path("assets/face_landmarker.task")

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MIN_VALID_JOINTS = 5


def check_model_files():
    missing = [str(path) for path in (POSE_MODEL, HAND_MODEL, FACE_MODEL) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing MediaPipe model file(s):\n- "
            + "\n- ".join(missing)
            + "\nCheck that all three .task files are inside the assets folder."
        )


# ==========================================
# 2. LANDMARK CONNECTIONS
# ==========================================

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FACE_CONTOURS = [
    [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
    [33, 160, 158, 133, 153, 144, 33],
    [362, 385, 387, 263, 373, 380, 362],
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61],
    [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 78],
    [70, 63, 105, 66, 107],
    [336, 296, 334, 293, 300],
]


# ==========================================
# 3. DRAWING HELPERS
# ==========================================

def get_point(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def landmark_is_visible(landmark, threshold=0.5):
    for attribute in ("visibility", "presence"):
        value = getattr(landmark, attribute, None)
        if value is not None:
            try:
                if float(value) < threshold:
                    return False
            except (TypeError, ValueError):
                pass
    return True


def draw_text(frame, text, position, scale=0.5, color=(255, 255, 255), thickness=1):
    cv2.putText(
        frame, str(text), position, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


def draw_pose(frame, result):
    height, width = frame.shape[:2]
    if not result.pose_landmarks:
        return False

    landmarks = result.pose_landmarks[0]
    points = [get_point(lm, width, height) for lm in landmarks]

    for start, end in POSE_CONNECTIONS:
        if not landmark_is_visible(landmarks[start]) or not landmark_is_visible(landmarks[end]):
            continue
        x1, y1 = points[start]
        x2, y2 = points[end]
        if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    for index, (x, y) in enumerate(points):
        if landmark_is_visible(landmarks[index]) and 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    return True


def classify_hand(landmarks):
    # Approximate upright-hand gesture heuristic; rotations can affect it.
    tips = (8, 12, 16, 20)
    pips = (6, 10, 14, 18)
    extended = sum(
        1 for tip, pip in zip(tips, pips)
        if landmarks[tip].y < landmarks[pip].y
    )
    if extended >= 3:
        return "OPEN HAND"
    if extended <= 1:
        return "FIST"
    return "PARTIAL HAND"


def draw_hands(frame, result):
    height, width = frame.shape[:2]
    for landmarks in result.hand_landmarks:
        points = [get_point(lm, width, height) for lm in landmarks]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 180, 0), 2)

        for x, y in points:
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        x, y = points[0]
        draw_text(
            frame, classify_hand(landmarks),
            (max(5, x - 40), max(25, y - 20)),
            0.55, (255, 180, 0), 2
        )

    return len(result.hand_landmarks)


def draw_face(frame, result):
    height, width = frame.shape[:2]
    if not result.face_landmarks:
        return False, "NO FACE"

    landmarks = result.face_landmarks[0]
    for contour in FACE_CONTOURS:
        points = [get_point(landmarks[index], width, height) for index in contour]
        cv2.polylines(
            frame, [np.asarray(points, dtype=np.int32)],
            False, (255, 0, 255), 1
        )

    for index in (1, 4, 10, 33, 133, 159, 145, 362, 263, 386, 374, 61, 291, 13, 14):
        x, y = get_point(landmarks[index], width, height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

    expression = "FACE DETECTED"
    if result.face_blendshapes:
        scores = {item.category_name: item.score for item in result.face_blendshapes[0]}
        smile = (scores.get("mouthSmileLeft", 0.0) + scores.get("mouthSmileRight", 0.0)) / 2.0
        blink = max(scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0))
        mouth_open = scores.get("jawOpen", 0.0)

        if blink > 0.65:
            expression = "BLINK"
        elif mouth_open > 0.65:
            expression = "MOUTH OPEN"
        elif smile > 0.55:
            expression = "SMILE"

    return True, expression


# ==========================================
# 4. POSE SELECTION
# ==========================================

POSE_KEYS = {
    ord("1"): "T-Pose",
    ord("2"): "Standing Pose",
    ord("3"): "Warrior Pose",
    ord("4"): "Squat Pose",
    ord("5"): "Tree Pose",
}

for pose_name in POSE_KEYS.values():
    if pose_name not in REFERENCE_POSES:
        raise ValueError(f"{pose_name!r} is missing from pose_matching.py")


# ==========================================
# 5. MAIN APPLICATION
# ==========================================

def main():
    check_model_files()

    BaseOptions = python.BaseOptions
    RunningMode = vision.RunningMode

    pose_detector = None
    hand_detector = None
    face_detector = None
    cap = None
    window_name = "SMART POSE AI - COMPLETE CAMERA"

    try:
        pose_options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        face_options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        face_detector = vision.FaceLandmarker.create_from_options(face_options)

        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            raise RuntimeError("Camera not detected. Check camera permission or CAMERA_INDEX.")

        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        screen_height = ctypes.windll.user32.GetSystemMetrics(1)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("SMART POSE AI STARTED!")
        print("Body + Hands + Face Tracking + Live Pose Matching")
        print("1=T-Pose | 2=Standing | 3=Warrior | 4=Squat | 5=Tree")
        print("Press Q or ESC to exit.")

        video_start_time = time.monotonic()
        last_timestamp = -1
        fps_window_start = time.monotonic()
        frame_count = 0
        fps = 0.0

        angle_smoother = AngleSmoother(alpha=0.35, max_missing_frames=5)
        selected_pose = "T-Pose"

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame error.")
                break

            # Mirror the display. MediaPipe landmark indexes remain anatomical left/right.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp = max(
                last_timestamp + 1,
                int((time.monotonic() - video_start_time) * 1000),
            )
            last_timestamp = timestamp

            # BODY DETECTION
            pose_result = pose_detector.detect_for_video(mp_image, timestamp)
            body_detected = draw_pose(frame, pose_result)

            # Use world landmarks for 3D joint estimates when available.
            raw_joint_angles = {name: None for name in JOINTS}
            if pose_result.pose_landmarks:
                if pose_result.pose_world_landmarks and len(pose_result.pose_world_landmarks) > 0:
                    angle_landmarks = pose_result.pose_world_landmarks[0]
                else:
                    angle_landmarks = pose_result.pose_landmarks[0]
                raw_joint_angles = extract_joint_angles(angle_landmarks)

            # Smoothed values are for display only when a joint disappears briefly.
            joint_angles = angle_smoother.update(raw_joint_angles)

            # CRITICAL FIX:
            # Match only joints detected in THIS frame. Do not score held/stale values.
            score_angles = {
                name: joint_angles[name] if raw_joint_angles.get(name) is not None else None
                for name in JOINTS
            }
            pose_match = compare_pose(
                score_angles,
                selected_pose,
                min_valid_joints=MIN_VALID_JOINTS,
            )

            # HAND DETECTION
            hand_result = hand_detector.detect_for_video(mp_image, timestamp)
            hand_count = draw_hands(frame, hand_result)

            # FACE DETECTION
            face_result = face_detector.detect_for_video(mp_image, timestamp)
            face_detected, expression = draw_face(frame, face_result)

            # DASHBOARD
            dashboard_height = 425
            cv2.rectangle(frame, (0, 0), (width, dashboard_height), (25, 25, 25), -1)

            body_status = "DETECTED" if body_detected else "NOT DETECTED"
            face_status = "DETECTED" if face_detected else "NOT DETECTED"

            draw_text(frame, "SMART POSE AI", (20, 30), 0.8, (255, 255, 255), 2)
            draw_text(
                frame, f"BODY: {body_status}", (20, 65), 0.55,
                (0, 255, 0) if body_detected else (0, 0, 255), 2
            )
            draw_text(frame, f"HANDS: {hand_count}/2", (300, 65), 0.55, (255, 180, 0), 2)
            draw_text(frame, f"FACE: {face_status}", (500, 65), 0.55, (255, 0, 255), 2)
            draw_text(frame, f"EXPRESSION: {expression}", (800, 65), 0.55, (0, 255, 255), 2)

            frame_count += 1
            elapsed = time.monotonic() - fps_window_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_window_start = time.monotonic()
            draw_text(frame, f"FPS: {fps:.1f}", (20, 105), 0.5, (255, 255, 255), 1)

            draw_text(frame, "JOINT ANGLES (3D estimate)", (20, 140), 0.55, (0, 255, 255), 2)

            for index, joint_name in enumerate(JOINTS):
                column = index // 4
                row = index % 4
                x_position = 20 + column * 350
                y_position = 175 + row * 23
                angle = joint_angles.get(joint_name)

                if raw_joint_angles.get(joint_name) is None:
                    # Clearly label a held value as temporarily missing.
                    if angle is None:
                        angle_text = f"{joint_name}: Not visible"
                    else:
                        angle_text = f"{joint_name}: {angle:.1f} deg (held)"
                else:
                    angle_text = f"{joint_name}: {angle:.1f} deg"

                draw_text(frame, angle_text, (x_position, y_position), 0.46, (0, 255, 255), 1)

            draw_text(frame, f"TARGET POSE: {selected_pose}", (20, 285), 0.58, (255, 255, 255), 2)

            if pose_match["ready"]:
                score = pose_match["score"]
                score_color = (0, 255, 0) if score >= 80 else (0, 200, 255)
                draw_text(
                    frame,
                    f"POSE MATCH: {score:.1f}% ({pose_match['valid_joints']}/8 current joints)",
                    (20, 320), 0.62, score_color, 2
                )

                feedback = pose_match["feedback"]
                message = feedback[0] if feedback else "All scored joints are within target ranges."
                draw_text(frame, f"FEEDBACK: {message[:82]}", (20, 355), 0.43, (0, 255, 255), 1)
            else:
                draw_text(
                    frame,
                    f"POSE MATCH: Waiting for valid joints ({pose_match['valid_joints']}/8)",
                    (20, 320), 0.55, (0, 165, 255), 2
                )
                draw_text(
                    frame,
                    f"Need at least {MIN_VALID_JOINTS} visible joints for a score.",
                    (20, 355), 0.43, (180, 180, 180), 1
                )

            draw_text(
                frame,
                "SELECT POSE: 1 T-Pose | 2 Standing | 3 Warrior | 4 Squat | 5 Tree",
                (20, 395), 0.43, (220, 220, 220), 1
            )
            draw_text(frame, "Q / ESC: EXIT", (width - 170, 395), 0.43, (220, 220, 220), 1)

            # Fullscreen display without stretching the camera image.
            scale = min(screen_width / width, screen_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))

            display = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            x_offset = (screen_width - new_width) // 2
            y_offset = (screen_height - new_height) // 2
            display[
                y_offset:y_offset + new_height,
                x_offset:x_offset + new_width
            ] = resized_frame

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in POSE_KEYS:
                selected_pose = POSE_KEYS[key]
                print(f"Selected target pose: {selected_pose}")

            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        if cap is not None:
            cap.release()
        if pose_detector is not None:
            pose_detector.close()
        if hand_detector is not None:
            hand_detector.close()
        if face_detector is not None:
            face_detector.close()
        cv2.destroyAllWindows()
        print("All AI camera resources closed.")


if __name__ == "__main__":
    main()
