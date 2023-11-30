import cv2
import mediapipe as mp
import time


class FaceMesh:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=10,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.f_mesh = mp.solutions.face_mesh
        self.face_mesh = self.f_mesh.FaceMesh(self.static_image_mode,
                                              self.max_num_faces,
                                              self.min_detection_confidence,
                                              self.min_tracking_confidence)
        self.draw_specs = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    def make_mesh(self, img, draw=True):
        f_lm_list = []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            for face_no, face_lms in enumerate(results.multi_face_landmarks):
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lms, self.f_mesh.FACE_CONNECTIONS, self.draw_specs, self.draw_specs)
                lm_list = []
                for id, lm in enumerate(face_lms.landmark):
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    lm_list.append([[id, x, y]])
                f_lm_list.append([face_no, lm_list])
        return f_lm_list


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    face_m = FaceMesh()
    while True:
        success, img = cap.read()
        pts = face_m.make_mesh(img)
        # print(pts)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 255, 255), 3)
        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
