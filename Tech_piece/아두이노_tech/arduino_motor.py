import serial

arduino = serial.Serial('COM6', 9600)  # 포트에 번호 맞추기


def motor(angle1, angle2):
    data = "{},{}\n".format(angle1, angle2)
    arduino.write(data.encode())


# 예제 실행
motor(90, 90)
