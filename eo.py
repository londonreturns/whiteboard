import cv2
import mediapipe as mp
import numpy as np
import math
import easyocr
import time

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def extract():
    gray_canvas = cv2.cvtColor(marker_canvas, cv2.COLOR_BGR2GRAY)
    _, threshold_canvas = cv2.threshold(gray_canvas, 0, 255, cv2.THRESH_BINARY)
    text = reader.readtext(threshold_canvas, detail=0, batch_size=5)
    cv2.imwrite("output.jpg", threshold_canvas)
    print("Recognized Text:", " ".join(text))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, static_image_mode=False, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils
reader = easyocr.Reader(['en'], gpu=True)

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
marker_thickness = 3

draw_buttons = True

button_colors = {
    "yellow": (0, 255, 255),
    "green": (0, 255, 0)
}

buttons = [
    ["Clear Screen", "yellow"],
    ["Text Extraction", "green"],
]

marker_colors = {
    "white": (255, 255, 255),
    "orange": (39, 132, 255),
    "purple": (215, 71, 143)
}

marker_color = marker_colors["white"]

marker_boxes = []

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
            cv2.rectangle(button_canvas, (axis_x, 10), (axis_x + width, 100), button_colors[button[1]], 5)
            cv2.putText(button_canvas, button[0], (axis_x + 18, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, button_colors[button[1]], 2)
            axis_x += gap + width

        gap, width = 115, 50
        axis_x, axis_y = 10, gap
        for name, color in marker_colors.items():
            box_start = (axis_x, axis_y)
            box_end = (axis_x + width, axis_y + 50)
            cv2.rectangle(button_canvas, box_start, box_end, marker_colors[name], -1)
            marker_boxes.append((box_start, box_end, marker_colors[name]))
            axis_y += 50 + 50

        draw_buttons = False

    if result is None:
        continue
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

                for box_start, box_end, color in marker_boxes:
                    if box_start[0] < index_finger_pos[0] < box_end[0] and \
                       box_start[1] < index_finger_pos[1] < box_end[1]:
                        marker_color = color

                if extract_button_start[0] < index_finger_pos[0] < extract_button_end[0] and \
                   extract_button_start[1] < index_finger_pos[1] < extract_button_end[1]:
                    if hover_start_time is None:
                        hover_start_time = current_time
                    elif current_time - hover_start_time >= 1:
                        if current_time - last_extract_time >= extract_buffer_time:
                            extract()
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

                if distance > 38:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(marker_canvas, (prev_x, prev_y), index_finger_pos, marker_color, marker_thickness)
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
        extract()

cap.release()
cv2.destroyAllWindows()