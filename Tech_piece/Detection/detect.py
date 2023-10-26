import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction


class Drone:
    def __init__(self):

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path='runs/detect/best2  .pt',
            confidence_threshold=0.3,
            device='cpu'
        )

        self.cap = cv2.VideoCapture('dji0088.mp4')

        self.tracker = None
        self.success = False

        #max tracking frame
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
            #Sliced inference
            #detection = get_sliced_prediction(frame, self.detection_model, slice_height=480, slice_width=480, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
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
                cprevx = self.prevx[:6]
                cprevy = self.prevy[:6]
                if max(cprevx) - min(cprevx) < 300 and max(cprevy) - min(cprevy) < 300 and len(cprevx) > 5:
                    roi = (xmin, ymin, xlen, ylen)
                    self.prevx = []
                    self.prevy = []
                self.tracker = cv2.TrackerCSRT_create()
                self.tracker.init(frame, roi)
                self.tframe = 0
            except Exception as e: 
                #print(e)
                self.tracker = None
                pass

        #tracking
        try:
            self.success, roi = self.tracker.update(frame)
            self.tframe += 1
            if self.success:
                (x, y, w, h) = tuple(map(int, roi))
                #cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if (x+w/2 < 5) or (x+w/2 > self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 5) or (y+h/2 < 5) or (y+h/2 > self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 5):
                    print('out of frame')
                    self.tracker = None
                loc = [x+w/2, y+h/2, self.label]
                return loc
            else:
                self.tracker = None
        except Exception as e:
            #print(e)
            pass


if __name__ == '__main__':

    start_command = input("Press 's' to start: ")

    if start_command == 's':
        drone = Drone()
        #drone.center()

        while True:
            re = drone.detect_and_find_center()
            #cv2.imshow('frame', drone.frame)
            print(re)
            if cv2.waitKey(1) == ord('q'):
                break
cv2.destroyAllWindows()