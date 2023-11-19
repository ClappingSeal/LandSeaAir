from ultralytics import YOLO

model = YOLO('runs/classify/train7/weights/best.pt')
detection = model('KakaoTalk_20231120_012342159.jpg', verbose=False)[0]
probs = list(detection.probs.data.tolist())
classes = detection.names
highest_prob = max(probs)
highest_prob_index = probs.index(highest_prob)
print(classes[highest_prob_index])