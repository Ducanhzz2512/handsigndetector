import pickle
import pyautogui
import cv2
import mediapipe as mp
import numpy as np
import time

# ===== LOAD MODEL =====
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# ===== LABELS A-Z =====
labels_dict = {i: chr(65 + i) for i in range(26)}

# ===== TEXT CONTROL =====
last_letter = ""
letter_start_time = 0
LETTER_DELAY = 2.0  # giữ 1 giây mới gõ

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

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # ===== FEATURE EXTRACT =====
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        # ===== PREDICT =====
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # ===== TEXT TYPING LOGIC =====
        current_time = time.time()

        if predicted_character == last_letter:
            if current_time - letter_start_time > LETTER_DELAY:
                pyautogui.write(predicted_character)
                print("Typed:", predicted_character)
                letter_start_time = current_time + 1  # tránh spam
        else:
            last_letter = predicted_character
            letter_start_time = current_time

        # ===== DISPLAY =====
        cv2.putText(frame, predicted_character,
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
