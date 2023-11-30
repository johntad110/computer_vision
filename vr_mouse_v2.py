import numpy as np
import time
import autopy
import cv2
import mediapipe as mp
import math


cam_w, cam_h = 640, 480
frame_r = 100
smoothening = 5

p_loc_x, p_loc_y = 0, 0
c_loc_x, p_Loc_y = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, cam_h)
cap.set(4, cam_h)
p_time = 0
w_scr, h_scr = autopy.screen.size()
print(w_scr, h_scr)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


def find_hands(img, draw=True):
    global results
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            if draw:
                mp_draw.draw_landmarks(img, hand_lm, mp_hands.HAND_CONNECTIONS)
    return img


def find_positions(img, hand_no=0, draw=True, draw_rec=True, draw_pt=None):
    global lm_list
    lm_list = []
    bbox = []
    x_list = []
    y_list = []
    if results.multi_hand_landmarks:
        my_hand = results.multi_hand_landmarks[hand_no]
        for id, lm in enumerate(my_hand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy])
            x_list.append(cx)
            y_list.append(cy)
            if draw & (draw_pt is None):
                cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)
            elif draw & (draw_pt is not None):
                if id in draw_pt:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)

        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)
        bbox = x_min, y_min, x_max, y_max

        if draw & draw_rec:
            cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 1)
    return lm_list, bbox


def fingers_up():
    fingers = []
    tip_id = [4, 8, 12, 16, 20]
    if lm_list[tip_id[0]][1] > lm_list[tip_id[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in range(1, 5):
        if lm_list[tip_id[id]][2] < lm_list[tip_id[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    print(fingers)
    return fingers


def find_distance(p1, p2, img, draw=True, r=5, t=3):
    x1, y1 = lm_list[p1][1:]
    x2, y2 = lm_list[p2][1:]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
    length = math.hypot(x2 - x1, y2 - y1)

    return length, img, [x1, y1, x2, y2, cx, cy]


while True:
    success, img = cap.read()
    img = find_hands(img)
    lm_list, bbox = find_positions(img, draw_pt=[])

    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        up_fingers = fingers_up()
        cv2.rectangle(img, (frame_r, frame_r), (cam_w - frame_r, cam_h - frame_r), (255, 0, 255), 1)

        if up_fingers[1] == 1 and up_fingers[2] == 0:
            x3 = np.interp(x1, (frame_r, cam_w - frame_r), (0, w_scr))
            y3 = np.interp(y1, (frame_r, cam_h - frame_r), (0, h_scr))
            c_loc_x = p_loc_x + (x3 - p_loc_x) / smoothening
            c_loc_y = p_loc_y + (y3 - p_loc_y) / smoothening
            autopy.mouse.move(w_scr - c_loc_x, c_loc_y)
            cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
            p_loc_x, p_loc_y = c_loc_x, c_loc_y
        if up_fingers[1] == 1 and up_fingers[2] == 1:
            length, img, line_info = find_distance(8, 12, img)
            print(length)
            if length < 30:
                cv2.circle(img, (line_info[4], line_info[5]), 7, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0, 255), 1)
    cv2.imshow("V Mouse", img)
    cv2.waitKey(1)
