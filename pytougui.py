import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_x = None
last_gesture_time = 0
COOLDOWN = 1.0   # seconds
THRESHOLD = 40   # pixels
gesture_text = ""   # Store text for display
gesture_display_time = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                # Center point (landmark 9 = middle finger MCP)
                cx = int(hand_landmark.landmark[9].x * w)

                # Draw center circle
                cv2.circle(frame, (cx, int(hand_landmark.landmark[9].y * h)), 
                           10, (0, 255, 0), -1)

                if prev_x is not None:
                    dx = cx - prev_x
                    current_time = time.time()

                    if abs(dx) > THRESHOLD and (current_time - last_gesture_time > COOLDOWN):
                        if dx > 0:
                            gesture_text = "👉 Swipe Right"
                            pyautogui.press("right")
                        else:
                            gesture_text = "👈 Swipe Left"
                            pyautogui.press("left")

                        gesture_display_time = current_time
                        last_gesture_time = current_time

                prev_x = cx
                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

        else:
            gesture_text = "No Hand Detected"
            gesture_display_time = time.time()

        # -------- Persistent Text for 1 second --------
        if time.time() - gesture_display_time < 1:
            cv2.putText(frame, gesture_text, (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Swipe Detection (Fixed)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
