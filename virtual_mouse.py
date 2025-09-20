import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Screen size
screen_w, screen_h = pyautogui.size()
last_click_time = 0
COOLDOWN = 0.7  # seconds

# To store last cursor pos (hold effect)
last_screen_x, last_screen_y = None, None

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

                # Get landmarks
                index_tip = hand_landmark.landmark[8]
                thumb_tip = hand_landmark.landmark[4]
                middle_tip = hand_landmark.landmark[12]
                index_base = hand_landmark.landmark[5]   # landmark 5 (index base)

                # Convert to pixel coords
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
                ibx, iby = int(index_base.x * w), int(index_base.y * h)

                # Distance between thumb & index base
                dist_thumb_indexbase = math.hypot(tx - ibx, ty - iby)

                # Convert to screen coords
                screen_x = int(index_tip.x * screen_w)
                screen_y = int(index_tip.y * screen_h)

                # ---- Movement only if thumb & index-base are close ----
                if dist_thumb_indexbase < 50:   # threshold
                    pyautogui.moveTo(screen_x, screen_y)
                    last_screen_x, last_screen_y = screen_x, screen_y
                    cv2.putText(frame, "MOVE", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                elif last_screen_x is not None and last_screen_y is not None:
                    # Hold position (no move)
                    pyautogui.moveTo(last_screen_x, last_screen_y)
                    cv2.putText(frame, "HOLD", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # ---- Distance checks for clicks ----
                dist_thumb_index = math.hypot(ix - tx, iy - ty)
                dist_thumb_middle = math.hypot(mx - tx, my - ty)

                current_time = time.time()

                # Left Click (Index + Thumb close)
                if dist_thumb_index < 40 and (current_time - last_click_time > COOLDOWN):
                    cv2.putText(frame, "Left Click", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    pyautogui.click()
                    last_click_time = current_time

                # Right Click (Middle + Thumb close)
                elif dist_thumb_middle < 40 and (current_time - last_click_time > COOLDOWN):
                    cv2.putText(frame, "Right Click", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    pyautogui.click(button='right')
                    last_click_time = current_time

                # Draw cursor pointer
                cv2.circle(frame, (ix, iy), 10, (0, 255, 255), -1)

        cv2.imshow("Virtual Mouse ", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
