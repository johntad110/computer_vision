import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
p_time = 0
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()


while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mp_draw.draw_detection(img, detection)
            h, w, c = img.shape
            bbox_c = detection.location_data.relative_bounding_box
            bbox = int(bbox_c.xmin * w), int(bbox_c.ymin * h), int(bbox_c.width * w), int(bbox_c.height * h)
            cv2.rectangle(img, bbox, (255, 244, 255), 2)
            cv2.putText(img, f"{int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (5, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("Vid", img)
    cv2.waitKey(1)
