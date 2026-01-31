import pickle
import pyautogui
import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ===== CONTROL VARS =====
last_click = 0
CLICK_DELAY = 0.5

prev_x, prev_y = 0, 0
smoothening = 5

last_letter = ""
letter_time = 0
LETTER_DELAY = 1.0  # giữ 1s để gõ

# ===== LOAD MODEL =====
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

labels_dict = {i: chr(65+i) for i in range(26)}

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

screen_w, screen_h = pyautogui.size()

# ===== FUNCTIONS =====
def distance(p1,p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

# ===== LOOP =====
while True:

    data_aux=[]
    x_=[]
    y_=[]

    ret, frame = cap.read()
    if not ret:
        break

    frame=cv2.flip(frame,1)

    H,W,_=frame.shape
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results=hands.process(rgb)

    if results.multi_hand_landmarks:

        hand=results.multi_hand_landmarks[0]

        # ===== MOUSE CONTROL =====
        index_finger=hand.landmark[8]
        thumb=hand.landmark[4]

        mouse_x=int(index_finger.x*screen_w)
        mouse_y=int(index_finger.y*screen_h)

        curr_x=prev_x+(mouse_x-prev_x)/smoothening
        curr_y=prev_y+(mouse_y-prev_y)/smoothening

        pyautogui.moveTo(curr_x,curr_y)
        prev_x,prev_y=curr_x,curr_y

        # click
        if distance(index_finger,thumb)<0.03:
            if time.time()-last_click>CLICK_DELAY:
                pyautogui.click()
                last_click=time.time()

        # ===== DRAW LANDMARKS =====
        mp_drawing.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

        # ===== ALPHABET RECOGNITION =====
        for lm in hand.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand.landmark:
            data_aux.append(lm.x-min(x_))
            data_aux.append(lm.y-min(y_))

        x1=int(min(x_)*W)-10
        y1=int(min(y_)*H)-10
        x2=int(max(x_)*W)-10
        y2=int(max(y_)*H)-10

        prediction=model.predict([np.asarray(data_aux)])
        predicted_character=labels_dict[int(prediction[0])]

        # ===== AUTO TYPE =====
        if predicted_character==last_letter:
            if time.time()-letter_time>LETTER_DELAY:
                pyautogui.typewrite(predicted_character)
                letter_time=time.time()+1
        else:
            last_letter=predicted_character
            letter_time=time.time()

        # ===== DISPLAY =====
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),4)
        cv2.putText(frame,predicted_character,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),3)

    cv2.imshow("Hand Control",frame)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
