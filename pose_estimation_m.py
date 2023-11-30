import cv2
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_Landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.static_image_mode,
                                      self.model_complexity,
                                      self.min_detection_confidence,
                                      self.min_tracking_confidence)

        self.mp_draw = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        res = self.results.pose_landmarks

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_points(self, img, draw=True, draw_points=None):
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw & (draw_points is None):
                    cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED)
                elif draw & (draw_points is not None):
                    if id in draw_points:
                        cx, cx = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED)
        return lm_list


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    while True:
        success, img = cap.read()
        detector = PoseDetector()
        img = detector.find_pose(img)
        my_list = detector.find_points(img, draw_points=[6, 2, 8])
        # print(my_list)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (5, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 3)
        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
