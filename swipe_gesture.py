import cv2
import mediapipe as mp
from collections import deque

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# History buffer for smooth swipe detection
history = deque(maxlen=5)  # last 5 hand centers

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # single hand for simplicity
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # selfie view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:

                h, w, _ = frame.shape
                # Hand center approx (Middle Finger MCP)
                cx = int(hand_landmark.landmark[9].x * w)
                cy = int(hand_landmark.landmark[9].y * h)

                # Add current center to history
                history.append((cx, cy))

                # Draw landmarks
                mp_draw.draw_landmarks(frame_bgr, hand_landmark, mp_hands.HAND_CONNECTIONS)

        # -------- Swipe Detection --------
        if len(history) >= 1:
            dx = history[-1][0] - history[0][0]
            dy = history[-1][1] - history[0][1]

            # Horizontal Swipe
            if abs(dx) > 80:
                if dx > 0:
                    cv2.putText(frame_bgr, "Swipe Right 👉", (50, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                else:
                    cv2.putText(frame_bgr, "Swipe Left 👈", (50, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

                history.clear()  # reset after swipe detection

            # Vertical Swipe
            elif abs(dy) > 80:
                if dy > 0:
                    cv2.putText(frame_bgr, "Swipe Down ⬇️", (50, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    cv2.putText(frame_bgr, "Swipe Up ⬆️", (50, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                history.clear()  # reset after swipe detection

        cv2.imshow("Smooth Swipe Detection", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
