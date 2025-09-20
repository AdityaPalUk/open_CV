import cv2
import mediapipe as mp

# Initialize Mediapipe
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,      
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip (selfie view)
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process
        results = hands.process(rgb_frame)

        # Back to BGR
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2)
                )

                # Finger detection logic
                tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
                fingers = []

                h, w, c = frame.shape

                for id in tip_ids:
                    if id != 4:  # Ignore thumb for now
                        if hand_landmark.landmark[id].y < hand_landmark.landmark[id-2].y:
                            fingers.append(1)  # finger up
                        else:
                            fingers.append(0)  # finger down
                    else:
                        fingers.append(0)  # ignoring thumb

                # ---- Gesture Detection ----
                if fingers == [0,1,1,0,0]:
                    cv2.putText(frame_bgr, "Peace ✌️", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                elif fingers == [0,0,0,0,0]:
                    cv2.putText(frame_bgr, "Fist ✊", (50,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                elif fingers == [0,1,1,1,1]:
                    cv2.putText(frame_bgr, "Open Palm 🖐️", (50,150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

                elif fingers == [0,1,0,0,0]:
                    cv2.putText(frame_bgr, "Pointing 👉", (50,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

                elif fingers == [0,1,0,0,1]:
                    cv2.putText(frame_bgr, "Rock 🤘", (50,250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)

                # ---- Special Case: OK Sign ----
                x1, y1 = int(hand_landmark.landmark[4].x * w), int(hand_landmark.landmark[4].y * h)  # Thumb tip
                x2, y2 = int(hand_landmark.landmark[8].x * w), int(hand_landmark.landmark[8].y * h)  # Index tip
                distance = ((x2-x1)**2 + (y2-y1)**2) ** 0.5

                # Dynamic threshold based on frame width
                threshold = w / 30  

                if distance < threshold:
                    cv2.putText(frame_bgr, "OK 👌", (50,300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
               



        # Show output
        cv2.imshow("Hand Tracking + Gesture", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
