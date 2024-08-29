import cv2
import mediapipe as mp
import numpy as np
import math
import easyocr

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def moving_average(points, window_size):
    if len(points) < window_size:
        return points
    return [tuple(np.mean(points[max(0, i-window_size+1):i+1], axis=0, dtype=int)) for i in range(len(points))]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = None, None
buffer = []
window_size = 5

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            h, w, c = frame.shape
            index_finger_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))

            distance = calculate_distance(index_finger_pos, thumb_pos)

            if distance < 23:
                if prev_x is not None and prev_y is not None:
                    buffer.append((prev_x, prev_y))
                    if len(buffer) > window_size:
                        buffer.pop(0)

                    smoothed_points = moving_average(buffer, window_size)
                    for i in range(1, len(smoothed_points)):
                        cv2.line(canvas, smoothed_points[i-1], smoothed_points[i], (255, 255, 255), 5)

                prev_x, prev_y = index_finger_pos
            else:
                prev_x, prev_y = None, None
                buffer = []

    else:
        prev_x, prev_y = None, None
        buffer = []

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