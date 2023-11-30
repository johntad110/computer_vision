import cv2
import time
import numpy as np
import hand_tracking_m as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


w_cam, h_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
p_time = 0

hand_detector = htm.HandDetector(min_detection_confidence=0.65)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
range = volume.GetVolumeRange()

min_vol = -65.25
max_vol = 0
vol = 0
vol_bar = 400
vol_per = 0
while True:
    success, img = cap.read()
    img = hand_detector.find_hands(img, draw=False)
    lm_list = hand_detector.find_positions(img, draw_pt=[4, 8], draw=True)
    if len(lm_list) != 0:
        # print(lm_list[4], lm_list[8])
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # hand range 200 - 26
        # volume range -65 - 0

        vol = np.interp(length, [26, 200], [min_vol, max_vol])
        vol_bar = np.interp(length, [26, 200], [400, 150])
        vol_per = np.interp(length, [26, 200], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        print(length, vol)

        if length < 50:
            cv2.circle(img, (cx, cy), 5, (25, 205, 165), cv2.FILLED)

    cv2.rectangle(img, (15, 150), (35, 400), (0, 255, 0), 1)
    cv2.rectangle(img, (15, int(vol_bar)), (35, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f"{int(vol_per)}%", (10, 425), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)
    cv2.imshow("Vid", img)
    cv2.waitKey(1)
