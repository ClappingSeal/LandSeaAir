import datetime
from ultralytics import YOLO
import cv2
import time

from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path


class Drone:
    def __init__(self):

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path='runs/detect/train/weights/best.pt',
            confidence_threshold=0.3,
            device='cpu'
        )

        self.cap = cv2.VideoCapture('dji0088.mp4')

#max tracking frame
        self.tracker = None
        self.success = False
        self.maxtrack = 180
        self.tframe = 0
        self.prevx = []
        self.prevy = []
        self.ret, self.frame = self.cap.read()
        self.label = None

    def detect_and_find_center(self):
        ret, frame = self.cap.read()
        conf = 0

        #cam check
        if not ret:
            print('Cam Error')
            return None

        #Detection
        if (self.tracker is None) or (self.tframe > self.maxtrack):
            detection = get_prediction(frame, self.detection_model)
            for data in detection.to_coco_annotations()[:3]:
                confidence = float(data['score'])
                if (confidence > conf) and (data['bbox'][2] < 100) and (data['bbox'][3] < 100):
                    xmin, ymin, xlen, ylen = int(data['bbox'][0]), int(data['bbox'][1]), int(data['bbox'][2]), int(data['bbox'][3])
                    xmid = xmin+xlen/2
                    ymid = ymin+ylen/2
                    conf = confidence
                    self.label = data['category_name']
            try:
                self.prevx.append(xmid)
                self.prevy.append(ymid)
                cprevx = self.prevx[:10]
                cprevy = self.prevy[:10]
                if max(cprevx) - min(cprevx) < 300 and max(cprevy) - min(cprevy) < 300 and len(cprevx) > 9:
                    roi = (xmin, ymin, xlen, ylen)
                    self.prevx = []
                    self.prevy = []
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(frame, roi)
                self.tframe = 0
            except:
                self.tracker = None
                pass

        #tracking
        try:
            self.success, roi = self.tracker.update(frame)
            self.tframe += 1
            if self.success:
                (x, y, w, h) = tuple(map(int, roi))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                loc = [x+w/2, y+h/2, self.label]
                return loc
            else:
                self.tracker = None
        except:
            pass


if __name__ == '__main__':

    start_command = input("Press 's' to start: ")

    if start_command == 's':
        drone = Drone()
        #drone.center()

        while True:
            re = drone.detect_and_find_center()
            print(re)
            if cv2.waitKey(1) == ord('q'):
                break
cv2.destroyAllWindows()
