import cv2
from ultralytics import YOLO


class Drone:
    def __init__(self):

        self.cap = cv2.VideoCapture(0)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
        self.tracker = None
        self.success = False
        self.model = YOLO('yolov5n300_full_integer_quant.tflite')
        #max tracking frame
        self.maxtrack = 180
        self.tframe = 0
        self.prevx = []
        self.prevy = []
        self.ret, self.frame = self.cap.read()
        self.label = None
        self.labels = ['fixed', 'quadcopter', 'hybrid', 'label']
    def detect_and_find_center(self):
        ret, self.frame = self.cap.read()
        conf = 0
        #self.frame = cv2.resize(self.frame, (1024, 768))
        #cam check
        if not ret:
            print('Cam Error')
            return None

        #Detection
        if (self.tracker is None) or (self.tframe > self.maxtrack):
            detection = self.model(self.frame, verbose=False, device='cpu', imgsz=1024)[0]
            #Sliced inference
            #detection = get_sliced_prediction(frame, self.detection_model, slice_height=480, slice_width=480, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
            for data in detection.boxes.data.tolist():
                confidence = float(data[4])
                xmin, ymin, xlen, ylen = int(data[0]), int(data[1]), int(data[2]) - int(data[0]), int(data[3]) - int(data[1])
                if (confidence > conf) and (xlen < 100) and (ylen < 100):
                    xmid = xmin+xlen/2
                    ymid = ymin+ylen/2
                    conf = confidence
                    self.label = self.labels[int(data[5])]
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
                self.tracker.init(self.frame, roi)
                self.tframe = 0
            except Exception as e: 
                print(e)
                self.tracker = None
                pass

        #tracking
        try:
            self.success, roi = self.tracker.update(self.frame)
            self.tframe += 1
            if self.success:
                (x, y, w, h) = tuple(map(int, roi))
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if (x+w/2 < 5) or (x+w/2 > self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 5) or (y+h/2 < 5) or (y+h/2 > self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 5):
                    print('out of frame')
                    self.tracker = None
                loc = [x+w/2, y+h/2, self.label]
                return loc
            else:
                self.tracker = None
        except Exception as e:
            print(e)
            pass


if __name__ == '__main__':

    start_command = input("Press 's' to start: ")

    if start_command == 's':
        drone = Drone()
        #drone.center()

        while True:
            re = drone.detect_and_find_center()
            cv2.imshow('frame', drone.frame)
            print(re)
            if cv2.waitKey(1) == ord('q'):
                break
cv2.destroyAllWindows()
