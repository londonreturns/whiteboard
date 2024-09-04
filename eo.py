import cv2
import mediapipe as mp
import numpy as np
import math
import easyocr
import time

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def extract(cv2):
    gray_canvas = cv2.cvtColor(marker_canvas, cv2.COLOR_BGR2GRAY)
    blurred_canvas = cv2.GaussianBlur(gray_canvas, (5, 5), 0)
    _, binary_canvas = cv2.threshold(blurred_canvas, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text = reader.readtext(binary_canvas, detail=0)
    print("Recognized Text:", " ".join(text))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.9)
mp_draw = mp.solutions.drawing_utils
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(0)

marker_canvas = None
button_canvas = None

prev_x, prev_y = None, None

marker_on = True
no_hands_start_time = None
one_hand_start_time = None
two_hands_start_time = None
two_hands_detected = False
hover_start_time = None
last_extract_time = 0
last_clear_time = 0

draw_buttons = True

colors = {
    "yellow": (0, 255, 255),
    "green": (0, 255, 0)
}

buttons = [
    ["Clear Screen", "yellow"],
    ["Text Extraction", "green"],
]

clear_button_start = (92, 10)
clear_button_end = (274, 100)

clear_buffer_time = 2

extract_button_start = (366, 10)
extract_button_end = (558, 100)

extract_buffer_time = 5

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)

    if marker_canvas is None:
        marker_canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    current_time = time.time()

    if draw_buttons and button_canvas is None:
        button_canvas = np.zeros_like(frame)
        gap, width = 92, 182
        axis_x, axis_y = gap, 0
        for i, button in enumerate(buttons):
            cv2.rectangle(button_canvas, (axis_x, 10), (axis_x + width, 100), colors[button[1]], 5)
            cv2.putText(button_canvas, button[0], (axis_x + 18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[button[1]], 2)
            axis_x += gap + width
        draw_buttons = False

    if result.multi_hand_landmarks:
        hand_count = len(result.multi_hand_landmarks)

        if hand_count == 1:
            if two_hands_detected:
                marker_on = True
                print("marker on after two hands")
            if one_hand_start_time is None:
                one_hand_start_time = current_time
            elif current_time - one_hand_start_time >= 5 and not two_hands_detected and not marker_on:
                marker_on = True
                print("marker on after 5 seconds")
            two_hands_detected = False
            two_hands_start_time = None
        elif hand_count == 2:
            marker_on = False
            prev_x, prev_y = None, None
            if not two_hands_detected:
                two_hands_detected = True
                two_hands_start_time = current_time
                marker_canvas = np.zeros_like(frame)
                print("two hands detected - canvas cleared")
        else:
            two_hands_detected = False
            one_hand_start_time = None
            two_hands_start_time = None

        no_hands_start_time = None

        if marker_on:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                h, w, c = frame.shape
                index_finger_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))

                distance = calculate_distance(index_finger_pos, thumb_pos)

                if extract_button_start[0] < index_finger_pos[0] < extract_button_end[0] and \
                   extract_button_start[1] < index_finger_pos[1] < extract_button_end[1]:
                    if hover_start_time is None:
                        hover_start_time = current_time
                    elif current_time - hover_start_time >= 1:
                        if current_time - last_extract_time >= extract_buffer_time:
                            extract(cv2)
                            last_extract_time = current_time
                            hover_start_time = None
                elif clear_button_start[0] < index_finger_pos[0] < clear_button_end[0] and \
                   clear_button_start[1] < index_finger_pos[1] < clear_button_end[1]:
                    if hover_start_time is None:
                        hover_start_time = current_time
                    elif current_time - hover_start_time >= 1:
                        if current_time - last_extract_time >= clear_buffer_time:
                            print("clear")
                            marker_canvas = np.zeros_like(frame)
                            last_extract_time = current_time
                            hover_start_time = None
                else:
                    hover_start_time = None

                if distance > 35:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(marker_canvas, (prev_x, prev_y), index_finger_pos, (255, 255, 255), 5)
                    prev_x, prev_y = index_finger_pos
                else:
                    prev_x, prev_y = None, None

    else:
        if no_hands_start_time is None:
            no_hands_start_time = current_time
        elif current_time - no_hands_start_time >= 5:
            if marker_on:
                marker_on = False
                print("marker off after no hands for 5 seconds")
        one_hand_start_time = None
        prev_x, prev_y = None, None

    frame = cv2.add(frame, marker_canvas)
    frame = cv2.add(frame, button_canvas)

    cv2.imshow("Whiteboard", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        marker_canvas = None
    elif key == ord('r'):
        extract(cv2)

cap.release()
cv2.destroyAllWindows()