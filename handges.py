import cv2
import mediapipe as mp
import numpy as np
import math

# ---------------- Setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)
# Standard 720p - Resize window manually if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

canvas = None
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
smoothing = 0.65 

colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]
current_color = colors[0]
dynamic_thickness = 10

def get_fingers_up(landmarks):
    tips = [8, 12, 16, 20]
    return [1 if landmarks[tip].y < landmarks[tip - 2].y else 0 for tip in tips]

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

while True:
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None: canvas = np.zeros((h, w, 3), np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # UI Header
    cv2.rectangle(frame, (0, 0), (w, 65), (30, 30, 30), -1)
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (i*140 + 20, 10), (i*140 + 130, 55), color, -1)
    
    cv2.putText(frame, f"Size: {dynamic_thickness}", (w-350, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        
        # 🟢 CALCULATE DYNAMIC THICKNESS
        # Distance between Wrist(0) and Middle Finger Mapped (9) or Thumb(4) to Pinky(20)
        # Using Wrist to Middle Finger base is usually more consistent for hand scale
        dist = calculate_distance(lm[0], lm[9]) 
        # Map distance (approx 0.1 to 0.5) to thickness (5 to 50)
        dynamic_thickness = int(np.interp(dist, [0.15, 0.45], [5, 60]))

        raw_x, raw_y = int(lm[8].x * w), int(lm[8].y * h)
        fingers = get_fingers_up(lm)

        # Smoothing EMA
        if curr_x == 0 and curr_y == 0:
            curr_x, curr_y = raw_x, raw_y
        else:
            curr_x = int(curr_x + (raw_x - curr_x) * smoothing)
            curr_y = int(curr_y + (raw_y - curr_y) * smoothing)

        # Drawing Mode (Index Up)
        if fingers[0] == 1 and fingers[1] == 0:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = curr_x, curr_y

            cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), 
                     current_color, dynamic_thickness)
            prev_x, prev_y = curr_x, curr_y
            # Feedback circle showing current thickness
            cv2.circle(frame, (curr_x, curr_y), dynamic_thickness // 2, (255, 255, 255), -1)

        # Selection Mode (Index + Middle Up)
        elif fingers[0] == 1 and fingers[1] == 1:
            prev_x, prev_y = 0, 0 
            if curr_y < 65:
                if 20 < curr_x < 130: current_color = colors[0]
                elif 160 < curr_x < 270: current_color = colors[1]
                elif 300 < curr_x < 410: current_color = colors[2]
                elif 440 < curr_x < 550: current_color = colors[3]
            cv2.circle(frame, (curr_x, curr_y), 15, current_color, 2)
        else:
            prev_x, prev_y = 0, 0

    # Blending
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Air Canvas (Press Q to Quit, C to Clear)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'): canvas = np.zeros((h, w, 3), np.uint8)

cap.release()
cv2.destroyAllWindows()
