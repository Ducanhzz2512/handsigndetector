import pickle
import pyautogui
import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ===== LOAD MODEL =====
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

# ===== LABELS A-Z =====
labels_dict = {i: chr(65 + i) for i in range(26)}

# ===== MOUSE CONTROL =====
screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0
smoothening = 5

# ===== SCROLL SETTINGS =====
last_scroll_y = None
SCROLL_SENSITIVITY = 40

# ===== DISTANCE FUNC =====
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# ===== MAIN LOOP =====
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # ===== MOUSE MOVE =====
        index_finger = hand_landmarks.landmark[8]
        middle_finger = hand_landmarks.landmark[12]

        mouse_x = int(index_finger.x * screen_w)
        mouse_y = int(index_finger.y * screen_h)

        curr_x = prev_x + (mouse_x - prev_x) / smoothening
        curr_y = prev_y + (mouse_y - prev_y) / smoothening

        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # ===== SCROLL MODE =====

        if distance(index_finger, middle_finger) < 0.04:

            curr_y = index_finger.y

            if last_scroll_y is not None:

                dy = curr_y - last_scroll_y

                # ngưỡng chống rung
                if abs(dy) > 0.01:
                    pyautogui.scroll(int(-dy * 3000))

            last_scroll_y = curr_y

        else:
            last_scroll_y = None


        # ===== DRAW LANDMARKS =====
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # ===== ALPHABET PREDICTION =====
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
