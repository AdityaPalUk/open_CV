import cv2
import mediapipe as mp

mp_face_det = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# MediaPipe expects min_detection_confidence in [0.0, 1.0]; using 0.5 is a safe default
face_dect = mp_face_det.FaceDetection(min_detection_confidence=0.5, model_selection=0)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    # Flip for mirror image
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB before processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_dect.process(frame_rgb)

    # Convert back to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Draw detections if any
    if result.detections:
        for det in result.detections:
            mp_draw.draw_detection(frame_bgr, det)

    # Show the frame
    cv2.imshow("Camera", frame_bgr)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


