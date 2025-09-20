import cv2
import mediapipe as mp
import pyautogui as pag  # Import the pyautogui library

# Initialize MediaPipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Initialize drawing utility for visualization
mp_drawing = mp.solutions.drawing_utils

def main():
    # Capture video from the default webcam
    cap = cv2.VideoCapture(0)
    
    # Get the screen resolution from pyautogui
    screen_width, screen_height = pag.size()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frame_rgb)

            if processed.multi_hand_landmarks:
                for hand_landmarks in processed.multi_hand_landmarks:
                    index_finger_tip = hand_landmarks.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
                    middle_finger_tip = hand_landmarks.landmark[mphands.HandLandmark.MIDDLE_FINGER_TIP]
                    
                    x_mouse = int(index_finger_tip.x * screen_width)
                    y_mouse = int(index_finger_tip.y * screen_height)

                    if abs(index_finger_tip.x - middle_finger_tip.x) < 0.05 and abs(index_finger_tip.y - middle_finger_tip.y) < 0.05:
                        pag.click()

                    pag.moveTo(x_mouse, y_mouse)

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mphands.HAND_CONNECTIONS)

            cv2.imshow("Virtual Mouse", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()