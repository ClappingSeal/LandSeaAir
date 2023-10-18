from pymavlink import mavutil
import cv2
import threading
import time
import serial
import struct


class Drone:
    def __init__(self, connection_string='/dev/ttyACM0', baudrate=115200):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.vehicle = mavutil.mavlink_connection(self.connection_string, baud=self.baudrate)
        self.camera = cv2.VideoCapture(0)
        self.is_recording = False
        self.out = None
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=3)
        self.crc16_tab = [0x0, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
                          0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
                          0x1231, 0x210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
                          0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
                          0x2462, 0x3443, 0x420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
                          0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
                          0x3653, 0x2672, 0x1611, 0x630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
                          0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
                          0x48c4, 0x58e5, 0x6886, 0x78a7, 0x840, 0x1861, 0x2802, 0x3823,
                          0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
                          0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0xa50, 0x3a33, 0x2a12,
                          0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
                          0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0xc60, 0x1c41,
                          0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
                          0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0xe70,
                          0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
                          0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
                          0x1080, 0xa1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
                          0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
                          0x2b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
                          0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
                          0x34e2, 0x24c3, 0x14a0, 0x481, 0x7466, 0x6447, 0x5424, 0x4405,
                          0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
                          0x26d3, 0x36f2, 0x691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
                          0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
                          0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x8e1, 0x3882, 0x28a3,
                          0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
                          0x4a75, 0x5a54, 0x6a37, 0x7a16, 0xaf1, 0x1ad0, 0x2ab3, 0x3a92,
                          0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
                          0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0xcc1,
                          0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
                          0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0xed1, 0x1ef0
                          ]

        self.center()

        if not self.camera.isOpened():
            print("Error: Couldn't open the camera.")
            return

        if self.ser.isOpen():
            print("Connection is established!")
        else:
            print("Error in serial connection!")

    ## 카메라 관련 함수

    def send_data(self, data):
        # Packing Data
        packed_data = bytearray()
        for item in data:
            packed_data += item.to_bytes(4, 'little')

        # 64byte Padding
        while len(packed_data) < 64:
            packed_data += b'\x00'

        # Sending Data
        self.vehicle.mav.data64_send(0, len(packed_data), packed_data)

    def show_camera_stream(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Couldn't read frame.")
                break

            if self.is_recording and self.out is not None:
                self.out.write(frame)

            cv2.imshow("Camera Stream", frame) # 이 부분 삭제 해야함
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # Press 'q' to quit
                break
            elif key == ord('s'):  # Press 's' to start recording
                if not self.is_recording:
                    self.out = cv2.VideoWriter('output.avi', fourcc, 20.0,
                                               (int(self.camera.get(3)), int(self.camera.get(4))))
                    self.is_recording = True
            elif key == ord('e'):  # Press 'e' to end recording
                if self.is_recording:
                    self.is_recording = False
                    if self.out is not None:
                        self.out.release()

        self.camera.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()

    ## 짐벌 카메라 동작 함수

    # rotate 과 center 함수 에서 사용됨
    def CRC16_cal(self, ptr, len_, crc_init=0):
        crc = crc_init
        for i in range(len_):
            temp = (crc >> 8) & 0xff
            crc = ((crc << 8) ^ self.crc16_tab[ptr[i] ^ temp]) & 0xffff
        return crc

    # CRC16_cal 함수 사용
    def rotate(self, x, y, t):
        cmd_header = b'\x55\x66\x01\x02\x00\x00\x00\x07'

        x_byte = struct.pack('b', x)
        y_byte = struct.pack('b', y)

        data_to_checksum = cmd_header + x_byte + y_byte
        calculated_checksum = self.CRC16_cal(data_to_checksum, len(data_to_checksum))
        checksum_bytes = struct.pack('<H', calculated_checksum)

        command = data_to_checksum + checksum_bytes

        self.ser.write(command)

        time.sleep(t)

        x_byte = struct.pack('b', 0)
        y_byte = struct.pack('b', 0)

        data_to_checksum = cmd_header + x_byte + y_byte
        calculated_checksum = self.CRC16_cal(data_to_checksum, len(data_to_checksum))
        checksum_bytes = struct.pack('<H', calculated_checksum)

        command = data_to_checksum + checksum_bytes

        self.ser.write(command)

    # CRC16_cal 함수 사용
    def center(self):
        cmd_header = b'\x55\x66\x01\x02\x00\x00\x00\x0E'

        x_byte = struct.pack('b', 0)
        y_byte = struct.pack('b', 0)

        data_to_checksum = cmd_header + x_byte + y_byte
        calculated_checksum = self.CRC16_cal(data_to_checksum, len(data_to_checksum))
        checksum_bytes = struct.pack('<H', calculated_checksum)

        command = data_to_checksum + checksum_bytes

        self.ser.write(command)
        time.sleep(2)

    def close_connection(self):
        self.vehicle.close()


if __name__ == '__main__':
    drone = Drone()

    camera_thread = threading.Thread(target=drone.show_camera_stream)
    camera_thread.start()

    drone.rotate(90, 90, 1)
    drone.center()

    while True:
        drone.send_data([123, 425, 234, 212])
        time.sleep(0.1)
