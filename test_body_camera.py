
import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL = "assets/pose_landmarker_lite.task"

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.35,
    min_pose_presence_confidence=0.35,
    min_tracking_confidence=0.35
)

detector = vision.PoseLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

start = time.monotonic()
timestamp = 0

try:
    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            print("Camera frame unavailable")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = max(
            timestamp + 1,
            int((time.monotonic() - start) * 1000)
        )

        result = detector.detect_for_video(image, timestamp)

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            print("\n--- BODY LANDMARK CONFIDENCE ---")
            for name, index in [
                ("Left Shoulder", 11),
                ("Right Shoulder", 12),
                ("Left Elbow", 13),
                ("Right Elbow", 14),
                ("Left Wrist", 15),
                ("Right Wrist", 16),
                ("Left Hip", 23),
                ("Right Hip", 24),
            ]:
                 point = landmarks[index]
                 print(name,"visibility:", getattr(point, "visibility", None),"presence:", getattr(point, "presence", None))
            visible = 0

            for a, b in CONNECTIONS:
                p1 = landmarks[a]
                p2 = landmarks[b]

                if p1.visibility > 0.35 and p2.visibility > 0.35:
                    x1, y1 = int(p1.x * width), int(p1.y * height)
                    x2, y2 = int(p2.x * width), int(p2.y * height)

                    cv2.line(frame, (x1, y1), (x2, y2),
                             (0, 255, 0), 3)

            for point in landmarks:
                if point.visibility > 0.35:
                    x = int(point.x * width)
                    y = int(point.y * height)

                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame, (x, y), 5,
                                   (0, 0, 255), -1)
                        visible += 1

            message = f"BODY DETECTED | LANDMARKS: {visible}"
            color = (0, 255, 0)

        else:
            message = "BODY NOT DETECTED"
            color = (0, 0, 255)

        cv2.putText(
            frame, message, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        cv2.imshow("Smart Pose - Body Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    detector.close()
    cv2.destroyAllWindows()
