import numpy as np
import cv2

# 저장된 내부 카메라 행렬과 왜곡 계수를 불러오기
mtx = np.load('camera_mtx.npy')
dist = np.load('camera_dist.npy')

cap = cv2.VideoCapture(0) # 기본 웹캠 사용

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 보정
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    cv2.imshow('Original', frame)
    cv2.imshow('Undistorted', undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
