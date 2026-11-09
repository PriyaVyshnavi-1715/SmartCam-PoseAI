
import cv2

# Open the laptop webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check camera connection
if not cap.isOpened():
    print("Error: Camera not detected!")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera started successfully!")
print("Press Q to close the camera.")

while True:

    # Read video frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot read camera frame!")
        break

    # Display information on frame
    cv2.putText(
        frame,
        "Computer Vision - Camera Test",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Display live video
    cv2.imshow("Smart Pose AI - Camera", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Camera closed successfully!")
