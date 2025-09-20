import cv2
import mediapipe as mp


mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)

# Initialize Hands Model

with mp_hands.Hands(
    static_image_mode=False,      # False for video stream
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for selfie view
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands
        results = hands.process(rgb_frame)

        # Convert back to BGR
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

       
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:

                # Draw points + connections
                mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2)
                )

                #  Finger Up Detection

                tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
                fingers = []

                h, w, c = frame.shape  # get frame size

                for id in tip_ids:
                    if id != 4:  # Ignore thumb for simplicity
                        # Tip y < lower joint y → finger up
                        if hand_landmark.landmark[id].y < hand_landmark.landmark[id-2].y:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    else:
                        fingers.append(0)  # Thumb ignore

                
                
                    


                # Gesture Detection
                if fingers == [0,1,1,0,0]:
                     cv2.putText(frame_bgr, "Peace", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                elif fingers == [0,0,0,0,0]:
                    cv2.putText(frame_bgr, "Fist ", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                elif fingers == [0,1,1,1,1]:
                    cv2.putText(frame_bgr, "Open Palm ", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)


        
        cv2.imshow("Hand Tracking + Gesture", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
