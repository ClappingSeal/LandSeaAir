import serial
import time


def send_command(ser, cmd):
    ser.write(cmd)
    response = ser.readline()
    return response


def set_gimbal_angle(ser, pitch, roll, yaw):
    cmd = "SET_ANGLES {} {} {}\n".format(pitch, roll, yaw).encode('utf-8')
    response = send_command(ser, cmd)
    return response


def main():
    # UART 설정
    ser = serial.Serial(
        port='/dev/ttyAMA1',
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
    )

    try:
        # 짐벌 카메라의 각도 설정 예시
        pitch = 30
        roll = 0
        yaw = 45
        response = set_gimbal_angle(ser, pitch, roll, yaw)
        print("Response:", response)

    finally:
        ser.close()


if __name__ == "__main__":
    main()
