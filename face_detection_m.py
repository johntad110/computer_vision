import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_confidence, self.model_selection)

    def detector(self, img, draw=True):
        bbox_r = []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)
        if results.detections:
            for id, detection in enumerate(results.detections):
                # self.mp_draw.draw_detection(img, detection)
                if draw:
                    h, w, c = img.shape
                    bbox_c = detection.location_data.relative_bounding_box
                    bbox = int(bbox_c.xmin * w), int(bbox_c.ymin * h), int(bbox_c.width * w), int(bbox_c.height * h)
                    bbox_r.append([id, bbox, detection.score])
                    cv2.rectangle(img, bbox, (255, 244, 255), 1)
                    img = self.draw_corner_box(img, bbox)
                    cv2.putText(img, f"{int(detection.score[0] * 100)}%",
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)
        return img, bbox_r

    def draw_corner_box(self, img, bbox, l=15, t=3):
        x, y, w, h = bbox
        x1, y1, = x+w, y+h
        # top left x, y
        cv2.line(img, (x, y), (x+l, y), (255, 255, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 255, 255), t)
        # top right x1, y
        cv2.line(img, (x1, y), (x1-l, y), (255, 255, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 255, 255), t)
        # bottom left x, y1
        cv2.line(img, (x, y1), (x+l, y1), (255, 255, 255), t)
        cv2.line(img, (x, y1), (x, y1-l), (255, 255, 255), t)
        # bottom right x1, y1
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 255, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 255, 255), t)
        return img


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    detect = FaceDetector()
    while True:
        success, img = cap.read()
        img, box = detect.detector(img)
        print(box)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, str(int(fps)), (5, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 3)
        cv2.imshow("Vid", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
