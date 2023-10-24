import datetime
from ultralytics import YOLO
import cv2
import time

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

model = YOLO('runs/detect/train/weights/best.pt')

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='runs/detect/train/weights/best.pt',
    confidence_threshold=0.3,
    device='cpu'
)

CONFIDENCE_THRESHOLD = 0.3
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
cap = cv2.VideoCapture('dji0088.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_rate = 60
prev = 0

tracker = None
success = True
tframe = 0
pxmid = 0
pymid = 0

prevx = []
prevy = []

labels = ['fixed', 'quadcopter', 'hybrid']
start = datetime.datetime.now()

while True:
    time_elapsed = time.time() - prev
    

    if time_elapsed > 1./frame_rate:
        ret, frame = cap.read()
        prev = time.time()
        conf = 0
 
        #cam check
        if not ret:
            print('Cam Error')
            break
            
        #Detection
        if (tracker is None) or (tframe > 180):
            detection = get_prediction(frame, detection_model)
            #detection = get_sliced_prediction(frame, detection_model, slice_height=480, slice_width=480, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
            for data in detection.to_coco_annotations()[:3]:
                confidence = float(data['score'])
                if (confidence > CONFIDENCE_THRESHOLD) and (confidence > conf) and (data['bbox'][2] < 100) and (data['bbox'][3] < 100):   
                    xmin, ymin, xlen, ylen = int(data['bbox'][0]), int(data['bbox'][1]), int(data['bbox'][2]), int(data['bbox'][3])
                    xmid = xmin+xlen/2
                    ymid = ymin+ylen/2
                    conf = confidence
                    label = data['category_name']
            try:
                prevx.append(xmid)
                prevy.append(ymid)

                cprevx = prevx[:10]
                cprevy = prevy[:10]

                if max(cprevx) - min(cprevx) < 300 and max(cprevy) - min(cprevy) < 300:
                    roi = (xmin, ymin, xlen, ylen)
                    prevx = []
                    prevy = []
                    print('roi selected')
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, roi)
                tframe = 0
                print('tracker reset')
            except:
                tracker = None
                pass

        #Draw output
        try:
            success, roi = tracker.update(frame)
            tframe += 1

            if success:
                (x, y, w, h) = tuple(map(int, roi))
                #print(x+w/2, y+h/2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                loc = [x+w/2, y+h/2, label]
                print(loc)
                cv2.putText(frame, label, (x, y), cv2.FONT_ITALIC, 0.5, WHITE, 2)
            else:
                tracker = None

            del confidence
            del xlen, xmin, ylen, ymin, roi, xmid, ymid
        except:
            pass


        end = datetime.datetime.now()

        total = (end - start).total_seconds()
        start = datetime.datetime.now()
        fps = f'FPS: {1 / total:.2f}'
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        cv2.imshow('Track', frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()