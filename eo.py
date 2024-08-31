import cv2
import mediapipe as mp
import numpy as np
import math
import easyocr
import time

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point1[1] - point2[1]) ** 2)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = None, None

marker_on = True
no_hands_start_time = None
one_hand_start_time = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    current_time = time.time()

    if result.multi_hand_landmarks:
        if len(result.multi_hand_landmarks) == 1:
            if one_hand_start_time is None:
                one_hand_start_time = current_time
            elif current_time - one_hand_start_time >= 5:
                if not marker_on:
                    marker_on = True
                    print("marker on")
        else:
            one_hand_start_time = None

        no_hands_start_time = None

        if marker_on:
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                h, w, c = frame.shape
                index_finger_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))

                distance = calculate_distance(index_finger_pos, thumb_pos)

                if distance > 23:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), index_finger_pos, (255, 255, 255), 5)
                    prev_x, prev_y = index_finger_pos
                else:
                    prev_x, prev_y = None, None

    else:
        if no_hands_start_time is None:
            no_hands_start_time = current_time
        elif current_time - no_hands_start_time >= 5:
            if marker_on:
                marker_on = False
                print("marker off")
        one_hand_start_time = None
        prev_x, prev_y = None, None

    frame = cv2.add(frame, canvas)

    cv2.imshow("Whiteboard", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        canvas = None
    elif key == ord('r'):
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        blurred_canvas = cv2.GaussianBlur(gray_canvas, (5, 5), 0)
        _, binary_canvas = cv2.threshold(blurred_canvas, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text = reader.readtext(binary_canvas, detail=0)
        print("Recognized Text:", " ".join(text))

cap.release()
cv2.destroyAllWindows()