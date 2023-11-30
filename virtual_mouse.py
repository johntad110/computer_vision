import numpy as np
import hand_tracking_m as htm
import time
import autopy
import cv2


cam_w, cam_h = 640, 480
frame_r = 100
smoothening = 5

p_loc_x, p_loc_y = 0, 0
c_loc_x, p_Loc_y = 0, 0


cap = cv2.VideoCapture(0)
cap.set(3, cam_h)
cap.set(4, cam_h)
p_time = 0
detector = htm.HandDetector(max_num_hands=1)
w_scr, h_scr = autopy.screen.size()
print(w_scr, h_scr)

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list, bbox = detector.find_positions(img, draw_pt=[] )

    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        fingers_up = detector.fingers_up()
        cv2.rectangle(img, (frame_r, frame_r), (cam_w - frame_r, cam_h - frame_r), (255, 0, 255), 1)

        if fingers_up[1] == 1 and fingers_up[2] == 0:
            x3 = np.interp(x1, (frame_r, cam_w - frame_r), (0, w_scr))
            y3 = np.interp(y1, (frame_r, cam_h - frame_r), (0, h_scr))
            c_loc_x = p_loc_x + (x3 - p_loc_x) / smoothening
            c_loc_y = p_loc_y + (y3 - p_loc_y) / smoothening
            autopy.mouse.move(w_scr - c_loc_x, c_loc_y)
            cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
            p_loc_x, p_loc_y = c_loc_x, c_loc_y
        if fingers_up[1] == 1 and fingers_up[2] == 1:
            length, img, line_info = detector.find_distance(8, 12, img)
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
