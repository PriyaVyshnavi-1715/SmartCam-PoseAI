import cv2
import mediapipe as mp
import time
import os
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from landmark_analysis import extract_joint_angles, AngleSmoother, JOINTS, PoseStabilityDetector
from pose_matching import REFERENCE_POSES, compare_pose
from sitting_posture import analyze_sitting_posture

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
POSE_MODEL = ASSETS_DIR / "pose_landmarker_lite.task"
HAND_MODEL = ASSETS_DIR / "hand_landmarker.task"
FACE_MODEL = ASSETS_DIR / "face_landmarker.task"
CAPTURES_DIR = PROJECT_ROOT / os.getenv("CAPTURES_DIR", "captures")

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FACE_CONTOURS = [
    [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10],
    [33,160,158,133,153,144,33], [362,385,387,263,373,380,362],
    [61,146,91,181,84,17,314,405,321,375,291,61],
    [78,191,80,81,82,13,312,311,310,415,308,78],
    [70,63,105,66,107], [336,296,334,293,300],
]

JOINT_INDEX_MAP = {
    11: "Left Shoulder", 12: "Right Shoulder",
    13: "Left Elbow", 14: "Right Elbow",
    15: "Left Elbow", 16: "Right Elbow",
    23: "Left Hip", 24: "Right Hip",
    25: "Left Knee", 26: "Right Knee",
    27: "Left Knee", 28: "Right Knee",
}


class PoseEngine:
    def __init__(self):
        self.camera = None
        self.pose_detector = None
        self.hand_detector = None
        self.face_detector = None
        self.selected_pose = "T-Pose"
        self.angle_smoother = AngleSmoother(alpha=0.35, max_missing_frames=5)
        self.stability_detector = PoseStabilityDetector(
            window_size=15, angle_threshold=5.0, required_stable_frames=10
        )
        self.running = False
        self.video_start_time = 0.0
        self.last_timestamp = -1
        self.latest_results = self._empty_results("Waiting for camera.")
        self.clean_frame = None
        self.latest_joint_scores = {}

    def _empty_results(self, message):
        return {
            "body_detected": False,
            "selected_pose": self.selected_pose,
            "match_ready": False,
            "match_score": None,
            "visible_joints": 0,
            "analysis_mode": "full_body",
            "required_joints": 5,
            "joint_angles": {name: None for name in JOINTS},
            "joint_scores": {name: None for name in JOINTS},
            "hand_count": 0,
            "hand_gestures": [],
            "face_detected": False,
            "expression": "NO FACE",
            "feedback": [message],
            "pose_stable": False,
            "sitting_analysis": {
                "ready": False,
                "shoulder_tilt_degrees": None,
                "head_offset": None,
                "feedback": ["Select Sitting Pose to begin."],
            },
        }



    def start(self):
        if self.running:
            return

        for model_path in (POSE_MODEL, HAND_MODEL, FACE_MODEL):
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Required MediaPipe model not found: {model_path}"
                )

        pose_detector = None
        hand_detector = None
        face_detector = None

        try:
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(POSE_MODEL)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            pose_detector = vision.PoseLandmarker.create_from_options(
                pose_options
            )

            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(HAND_MODEL)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            hand_detector = vision.HandLandmarker.create_from_options(
                hand_options
            )

            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(FACE_MODEL)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
            )
            face_detector = vision.FaceLandmarker.create_from_options(
                face_options
            )

            self.pose_detector = pose_detector
            self.hand_detector = hand_detector
            self.face_detector = face_detector

            self.camera = None
            self.angle_smoother.reset()
            self.stability_detector.reset()
            self.video_start_time = time.monotonic()
            self.last_timestamp = -1
            self.running = True

            self.latest_results = self._empty_results(
                "Move into the camera view."
            )

            print("SMART CAMERA ENGINE STARTED")

        except Exception:
            for detector in (
                pose_detector,
                hand_detector,
                face_detector,
            ):
                if detector is not None:
                    detector.close()
            raise

    def stop(self):
        self.running = False

        for attr in (
            "pose_detector",
            "hand_detector",
            "face_detector",
        ):
            detector = getattr(self, attr)

            if detector is not None:
                detector.close()
                setattr(self, attr, None)

        self.camera = None
        self.angle_smoother.reset()
        self.stability_detector.reset()
        self.clean_frame = None
        self.latest_results = self._empty_results("Camera stopped.")

        print("SMART CAMERA ENGINE STOPPED")

    def set_target_pose(self, pose_name):
        if pose_name not in REFERENCE_POSES:
            raise ValueError(f"Invalid pose: {pose_name}")
        self.selected_pose = pose_name
        return {"selected_pose": self.selected_pose}

    def _get_joint_color(self, landmark_index):
        joint_name = JOINT_INDEX_MAP.get(landmark_index)
        if joint_name is None:
            return (0, 0, 255)
        score = self.latest_joint_scores.get(joint_name)
        if score is None:
            return (0, 0, 255)
        if score >= 80:
            return (0, 255, 0)
        if score >= 50:
            return (0, 255, 255)
        return (0, 0, 255)

    def draw_pose(self, frame, landmarks):
        height, width = frame.shape[:2]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        for start, end in POSE_CONNECTIONS:
            first, second = landmarks[start], landmarks[end]
            if getattr(first, "visibility", 1.0) < 0.35 or getattr(second, "visibility", 1.0) < 0.35:
                continue
            x1, y1 = points[start]
            x2, y2 = points[end]
            if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        for index, (x, y) in enumerate(points):
            if getattr(landmarks[index], "visibility", 1.0) >= 0.35 and 0 <= x < width and 0 <= y < height:
                color = self._get_joint_color(index)
                cv2.circle(frame, (x, y), 7, color, -1)
                cv2.circle(frame, (x, y), 9, (255, 255, 255), 1)

    def draw_alignment_rings(self, frame, landmarks):
        height, width = frame.shape[:2]
        for index, name in JOINT_INDEX_MAP.items():
            if index >= len(landmarks):
                continue
            lm = landmarks[index]
            if getattr(lm, "visibility", 1.0) < 0.35:
                continue
            x, y = int(lm.x * width), int(lm.y * height)
            if not (0 <= x < width and 0 <= y < height):
                continue
            score = self.latest_joint_scores.get(name)
            if score is not None and score < 80:
                ring_color = (0, 255, 255) if score >= 50 else (0, 0, 255)
                cv2.circle(frame, (x, y), 14, ring_color, 2)

    @staticmethod
    def classify_hand(landmarks):
        pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        extended = sum(1 for tip, pip in pairs if landmarks[tip].y < landmarks[pip].y)
        if extended >= 3:
            return "OPEN HAND"
        if extended <= 1:
            return "FIST"
        return "PARTIAL HAND"

    def draw_hands(self, frame, hand_result):
        height, width = frame.shape[:2]
        gestures = []
        for landmarks in hand_result.hand_landmarks:
            points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], (255, 180, 0), 2)
            for x, y in points:
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            gesture = self.classify_hand(landmarks)
            gestures.append(gesture)
            wx, wy = points[0]
            cv2.putText(frame, gesture, (wx, max(25, wy - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)
        return gestures

    def draw_face(self, frame, face_result):
        if not face_result.face_landmarks:
            return False, "NO FACE"
        height, width = frame.shape[:2]
        landmarks = face_result.face_landmarks[0]
        for contour in FACE_CONTOURS:
            for i in range(len(contour) - 1):
                a, b = landmarks[contour[i]], landmarks[contour[i + 1]]
                p1 = (int(a.x * width), int(a.y * height))
                p2 = (int(b.x * width), int(b.y * height))
                if 0 <= p1[0] < width and 0 <= p1[1] < height and 0 <= p2[0] < width and 0 <= p2[1] < height:
                    cv2.line(frame, p1, p2, (255, 0, 255), 1)
        expression = "FACE DETECTED"
        if face_result.face_blendshapes:
            scores = {item.category_name: item.score for item in face_result.face_blendshapes[0]}
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

    def capture_frame(self):
        """Capture the current clean frame and save to captures directory."""
        if self.clean_frame is None:
            return None
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = CAPTURES_DIR / filename
        cv2.imwrite(str(filepath), self.clean_frame)
        results = self.latest_results
        return {
            "image_path": str(filepath),
            "filename": filename,
            "alignment_score": results.get("match_score"),
            "pose_name": results.get("selected_pose"),
            "pose_stable": results.get("pose_stable", False),
        }

    def process_frame(self, frame):
        if not self.running or any(
            detector is None
            for detector in (
                self.pose_detector,
                self.hand_detector,
                self.face_detector,
            )
        ):
            raise RuntimeError("Pose engine and models are not running.")

        if frame is None:
            raise ValueError("No camera frame received.")

        frame = cv2.flip(frame.copy(), 1)
        self.clean_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp = max(
            self.last_timestamp + 1,
            int((time.monotonic() - self.video_start_time) * 1000),
        )
        self.last_timestamp = timestamp

        pose_result = self.pose_detector.detect_for_video(
            mp_image, timestamp
        )

        body_detected = bool(pose_result.pose_landmarks)

        raw_angles = {name: None for name in JOINTS}
        stable_angles = {name: None for name in JOINTS}

        sitting_analysis = {
            "ready": False,
            "shoulder_tilt_degrees": None,
            "head_offset": None,
            "feedback": ["Select Sitting Pose to begin."],
        }

        pose_stable = False
        hand_count = 0
        hand_gestures = []
        face_detected = False
        expression = "NO FACE"

        pose_match = {
            "ready": False,
            "score": None,
            "valid_joints": 0,
            "required_joints": 5,
            "feedback": ["Move into camera view."],
            "joint_scores": {},
            "analysis_mode": "full_body",
        }

        if body_detected:
            landmarks = pose_result.pose_landmarks[0]

            self.draw_pose(frame, landmarks)

            angle_landmarks = (
                pose_result.pose_world_landmarks[0]
                if pose_result.pose_world_landmarks
                else landmarks
            )

            raw_angles = extract_joint_angles(
                angle_landmarks,
                confidence_landmarks=landmarks,
                threshold=0.35,
            )

            if self.selected_pose == "Sitting Pose":
                sitting_analysis = analyze_sitting_posture(landmarks)

            stable_angles = self.angle_smoother.update(raw_angles)

            score_angles = {
                name: (
                    stable_angles[name]
                    if raw_angles.get(name) is not None
                    else None
                )
                for name in JOINTS
            }

            pose_match = compare_pose(
                score_angles,
                self.selected_pose,
            )

            self.latest_joint_scores = pose_match.get(
                "joint_scores", {}
            )

            self.draw_alignment_rings(frame, landmarks)

            pose_stable = self.stability_detector.update(raw_angles)

            hand_result = self.hand_detector.detect_for_video(
                mp_image, timestamp
            )
            hand_gestures = self.draw_hands(frame, hand_result)
            hand_count = len(hand_gestures)

            face_result = self.face_detector.detect_for_video(
                mp_image, timestamp
            )
            face_detected, expression = self.draw_face(
                frame, face_result
            )

            cv2.putText(
                frame,
                f"TARGET: {self.selected_pose}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if pose_match["ready"]:
                score_text = (
                    f"POSE MATCH: {pose_match['score']:.1f}%"
                )
                feedback = pose_match["feedback"]
                feedback_text = (
                    feedback[0]
                    if feedback
                    else "All visible joints are within target ranges."
                )
            else:
                score_text = (
                    f"WAITING FOR JOINTS: "
                    f"{pose_match['valid_joints']}/"
                    f"{pose_match['required_joints']}"
                )
                feedback = pose_match["feedback"]
                feedback_text = (
                    feedback[0]
                    if feedback
                    else "Move into camera view."
                )

            score_color = (0, 255, 0)

            if pose_match["ready"] and pose_match["score"] is not None:
                if pose_match["score"] < 50:
                    score_color = (0, 0, 255)
                elif pose_match["score"] < 80:
                    score_color = (0, 255, 255)

            cv2.putText(
                frame,
                score_text,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                score_color,
                2,
            )

            cv2.putText(
                frame,
                feedback_text[:75],
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
            )

            if (
                pose_stable
                and pose_match["ready"]
                and pose_match["score"] is not None
                and pose_match["score"] > 70
            ):
                cv2.putText(
                    frame,
                    "POSE STABLE - Ready for capture!",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

            hand_text = (
                ", ".join(hand_gestures)
                if hand_gestures
                else "NO HANDS"
            )

            cv2.putText(
                frame,
                f"HANDS ({hand_count}/2): {hand_text}"[:75],
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 180, 0),
                1,
            )

            cv2.putText(
                frame,
                f"FACE: {'DETECTED' if face_detected else 'NOT DETECTED'}",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 0, 255),
                1,
            )

            cv2.putText(
                frame,
                f"EXPRESSION: {expression}",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
            )

        visible_joints = sum(
            value is not None for value in raw_angles.values()
        )

        results = {
            "body_detected": body_detected,
            "selected_pose": self.selected_pose,
            "match_ready": pose_match["ready"],
            "match_score": pose_match["score"],
            "visible_joints": visible_joints,
            "analysis_mode": pose_match.get(
                "analysis_mode", "full_body"
            ),
            "required_joints": pose_match.get(
                "required_joints", 5
            ),
            "joint_angles": {
                name: (
                    round(value, 1)
                    if value is not None
                    else None
                )
                for name, value in stable_angles.items()
            },
            "joint_scores": pose_match.get("joint_scores", {}),
            "hand_count": hand_count,
            "hand_gestures": hand_gestures,
            "face_detected": face_detected,
            "expression": expression,
            "feedback": pose_match["feedback"],
            "pose_stable": pose_stable,
            "sitting_analysis": sitting_analysis,
        }

        self.latest_results = results

        return frame, results
