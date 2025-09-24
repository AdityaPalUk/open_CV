import cv2
import mediapipe as mp
import os

# using MediaPipe Face Detection
mp_face_det = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_dect = mp_face_det.FaceDetection(min_detection_confidence=0.7, model_selection=0)

# Create folder to save faces
if not os.path.exists("faces"):
    os.makedirs("faces")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

count = 0       # Number of captured images
face_present = False  # Is a face currently detected

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_dect.process(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if result.detections:
        face_present = True
        for det in result.detections:
            mp_draw.draw_detection(frame_bgr, det)
        cv2.putText(frame_bgr, f"Press 'S' to Save Face 📸 (Captured: {count})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        face_present = False
        cv2.putText(frame_bgr, "No Face Detected ❌", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Resize the frame to make window smaller 
    frame_small = cv2.resize(frame_bgr, (0, 0), fx=0.7, fy=0.7)
    cv2.imshow("Camera", frame_small)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and face_present:
        count += 1
        filename = f"faces/face_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Face saved: {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
