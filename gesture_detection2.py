import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2)
                )

                h, w, c = frame.shape  # get frame size

                # ---------- Thumbs Up / Down Detection ----------
                thumb_tip  = hand_landmark.landmark[4]   # Thumb tip
                thumb_ip   = hand_landmark.landmark[3]   # Thumb joint
                index_mcp  = hand_landmark.landmark[5]   # Index base (reference point)

                # Convert to pixel coords (only Y needed here)
                thumb_tip_y  = int(thumb_tip.y * h)
                index_mcp_y  = int(index_mcp.y * h)

                # Check other fingers (index, middle, ring, pinky should be down)
                other_fingers = []
                for i in [8, 12, 16, 20]:
                    if hand_landmark.landmark[i].y < hand_landmark.landmark[i-2].y:
                        other_fingers.append(1)  # finger up
                    else:
                        other_fingers.append(0)  # finger down

                # If all other fingers are down → check thumb orientation
                if sum(other_fingers) == 0:
                    if thumb_tip_y < index_mcp_y:  
                        # Thumb above index base → Thumbs Up
                        cv2.putText(frame_bgr, "Thumbs Up 👍", (50, 350),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    elif thumb_tip_y > index_mcp_y:  
                        # Thumb below index base → Thumbs Down
                        cv2.putText(frame_bgr, "Thumbs Down 👎", (50, 350),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Hand Tracking + Gesture", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
