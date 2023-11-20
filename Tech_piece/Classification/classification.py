from ultralytics import YOLO
import cv2

class Drone:
    def __init__(self):    
        self.model = YOLO('best.pt')
        self.frame = cv2.imread('KakaoTalk_20231120_012342159.jpg')
    def detect2(self, xmid, ymid, xlen, ylen, frame):
        h, w , _ = frame.shape
        xmin = int(xmid-xlen/2) if int(xmid-xlen/2) >= 0 else 0
        xmax = int(xmid+xlen/2) if int(xmid+xlen/2) <= w else w
        ymin = int(ymid-ylen/2) if int(ymid-ylen/2) >= 0 else 0
        ymax = int(ymid+ylen/2) if int(ymid+ylen/2) <= h else h
        cframe = frame[ymin:ymax, xmin:xmax]
        detection = self.model(cframe, verbose=False)[0]
        probs = list(detection.probs.data.tolist())
        classes = detection.names
        highest_prob = max(probs)
        highest_prob_index = probs.index(highest_prob)
        return classes[highest_prob_index]


drone = Drone()
print(drone.detect2(100,100,1000,1000,drone.frame))