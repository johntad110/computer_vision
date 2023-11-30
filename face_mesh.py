import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
p_time = 0

mp_draw = mp.solutions.drawing_utils
f_mesh = mp.solutions.face_mesh
face_mesh = f_mesh.FaceMesh(max_num_faces=10)
draw_specs = mp_draw.DrawingSpec(thickness=1, circle_radius=1)


while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        for face_no, face_lms in enumerate(results.multi_face_landmarks):
            mp_draw.draw_landmarks(img, face_lms, f_mesh.FACE_CONNECTIONS, draw_specs, draw_specs)
            for id, lm in enumerate(face_lms.landmark):
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("Video", img)
    cv2.waitKey(1)
