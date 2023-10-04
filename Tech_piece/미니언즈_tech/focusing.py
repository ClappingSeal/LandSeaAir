import cv2
import numpy as np
import serial
import time

# Arduino에 연결
arduino = serial.Serial('COM3', 9600)

def motor(angle1, angle2):
    data = "{},{}\n".format(angle1, angle2)
    arduino.write(data.encode())

# 비디오 파일 또는 카메라 인덱스를 지정
cap = cv2.VideoCapture(0)

# 카메라 해상도를 320x240으로 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
angle1 = 90
angle2 = 90

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # 카메라 중앙 좌표
    center_x = width // 2
    center_y = height // 2

    if not ret:
        print("비디오를 불러올 수 없습니다.")
        break

    # BGR에서 HSV로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 노란색의 HSV 범위를 정의
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # 노란색 마스크 생성
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 노란색 마스크로 노란색 부분만 추출
    yellow = cv2.bitwise_and(frame, frame, mask=mask)

    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                dx = center_x - cx
                dy = center_y - cy

                angle1 += dx * 0.03
                angle2 += dy * 0.03

                time.sleep(0.05)
                motor(int(angle1), int(angle2))

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                print(f"중심점 좌표: ({cx}, {cy})")

    # 결과 화면 표시
    cv2.imshow('frame', frame)
    cv2.imshow('yellow', yellow)  # 빨간색 대신 노란색 이미지를 표시

    # q 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
