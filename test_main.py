from dronekit import connect
import cv2
import time
import socket
import struct
import logging
import numpy as np
import os
import math
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class Drone:
    def __init__(self, connection_string='/dev/ttyACM0', baudrate=115200, udp_ip="0.0.0.0", udp_port=37260):

        # Connecting value
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.vehicle = connect(self.connection_string, wait_ready=False, baud=self.baudrate, timeout=100)

        # Communication
        self.received_data = None
        self.vehicle.add_message_listener('DATA64', self.data64_callback)

        # Camera
        self.camera = cv2.VideoCapture(0)

        # Camera_color_test1
        self.ret, self.frame = self.camera.read()
        self.base_color = np.array([100, 255, 255])
        self.image_count = 0
        self.threshold = 10
        self.alpha = 0.3

        # Gimbal UDP Settings
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.udp_socket.bind((self.udp_ip, self.udp_port))
            print("UDP socket bound successfully!")
        except socket.error as e:
            print(f"Error binding UDP socket: {e}")
            return
        # Gimbal
        # self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=3)
        self.current_yaw = 0
        self.current_pitch = -90
        self.frame_width = 850
        self.frame_height = 480
        self.max_yaw = 10
        self.max_pitch = 10
        self.min_pitch = -10
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

        if not self.camera.isOpened():
            print("Error: Couldn't open the camera.")
            return

    # drone detect camera frame
    def detect_and_find_center(self):
        ret, frame = self.camera.read()
        conf = 0
    
        # cam check
        if not ret:
            print('Cam Error')
            return None
    
        # Detection
        if (self.tracker is None) or (self.tframe > self.maxtrack):
            detection = get_prediction(frame, self.detection_model)
            # Sliced inference
            # detection = get_sliced_prediction(frame, self.detection_model, slice_height=480, slice_width=480, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
            for data in detection.to_coco_annotations()[:3]:
                confidence = float(data['score'])
                if (confidence > conf) and (data['bbox'][2] < 100) and (data['bbox'][3] < 100):
                    xmin, ymin, xlen, ylen = int(data['bbox'][0]), int(data['bbox'][1]), int(data['bbox'][2]), int(
                        data['bbox'][3])
                    xmid = xmin + xlen / 2
                    ymid = ymin + ylen / 2
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
                # print(e)
                self.tracker = None
                pass
    
        # tracking
        try:
            self.success, roi = self.tracker.update(frame)
            self.tframe += 1
            if self.success:
                (x, y, w, h) = tuple(map(int, roi))
                # cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if (x + w / 2 < 5) or (x + w / 2 > self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT) - 5) or (y + h / 2 < 5) or (
                        y + h / 2 > self.camera.get(cv2.CAP_PROP_FRAME_WIDTH) - 5):
                    print('out of frame')
                    self.tracker = None
                loc = [x + w / 2, y + h / 2, self.label]
                print(loc)
                return loc
            else:
                self.tracker = None
        except Exception as e:
            # print(e)
            pass
        
    # Receiving 1
    def data64_callback(self, vehicle, name, message):
        # Unpacking the received data
        data = [int.from_bytes(message.data[i:i + 4], 'little') for i in range(0, len(message.data), 4)]
        self.received_data = data

    # Receiving 2
    def receiving_data(self):
        return self.received_data

    # Transmitting
    def sending_data(self, data):
        # Packing Data
        packed_data = bytearray()
        for item in data:
            packed_data += item.to_bytes(4, 'little')

        while len(packed_data) < 64:
            packed_data += b'\x00'

        msg = self.vehicle.message_factory.data64_encode(0, len(packed_data), packed_data)
        self.vehicle.send_mavlink(msg)

    # gimbal 1
    def CRC16_cal(self, ptr, len_, crc_init=0):
        crc = crc_init
        for i in range(len_):
            temp = (crc >> 8) & 0xff
            crc = ((crc << 8) ^ self.crc16_tab[ptr[i] ^ temp]) & 0xffff
        return crc

    # gimbal 2
    def set_gimbal_angle(self, yaw, pitch):  # 각도 체크섬 생성 및 각도 조종 명령 주기
        cmd_header = b'\x55\x66\x01\x04\x00\x00\x00\x0E'
        yaw_bytes = struct.pack('<h', int(yaw * 10))
        pitch_bytes = struct.pack('<h', int(pitch * 10))
        data_to_checksum = cmd_header + yaw_bytes + pitch_bytes
        calculated_checksum = self.CRC16_cal(data_to_checksum, len(data_to_checksum))
        checksum_bytes = struct.pack('<H', calculated_checksum)
        command = data_to_checksum + checksum_bytes
        self.send_command_to_gimbal(command)

        self.current_yaw = yaw
        self.current_pitch = pitch

    # gimbal 3
    def send_command_to_gimbal(self, command_bytes):
        try:
            self.udp_socket.sendto(command_bytes, ("192.168.144.25", self.udp_port))
            # print("Command sent successfully!")
        except socket.error as e:
            print(f"Error sending command via UDP: {e}")

    # gimbal 4
    def adjust_gimbal_relative_to_current(self, target_x, target_y):  # 상대 각도
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2

        diff_x = target_x - center_x
        diff_y = target_y - center_y

        yaw_adjustment = self.current_yaw + diff_x
        pitch_adjustment = self.current_pitch - diff_y

        # yaw_adjustment = max(-self.max_yaw, min(self.max_yaw, yaw_adjustment))
        # pitch_adjustment = max(self.min_pitch, min(self.max_pitch, pitch_adjustment))

        self.set_gimbal_angle(-yaw_adjustment, pitch_adjustment)
        print(target_x, target_y)
        print(yaw_adjustment, -pitch_adjustment)

    # gimbal 5
    def adjust_gimbal(self, target_x, target_y):  # 절대 각도
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2

        diff_x = target_x - center_x
        diff_y = target_y - center_y

        scale_factor_yaw = 135 / center_x
        scale_factor_pitch = (25 + 90) / center_y

        yaw_adjustment = self.current_yaw + diff_x * scale_factor_yaw
        pitch_adjustment = self.current_pitch - diff_y * scale_factor_pitch

        yaw_adjustment = max(-135, min(135, yaw_adjustment))
        pitch_adjustment = max(-90, min(25, pitch_adjustment))

        self.set_gimbal_angle(yaw_adjustment, pitch_adjustment)

    def close_connection(self):
        self.vehicle.close()

    def images_to_avi(self, image_prefix, base_output_filename, fps=10):
        files = os.listdir()
        jpg_files = [file for file in files if file.startswith(image_prefix) and file.endswith('.jpg')]

        jpg_files.sort(key=lambda x: int(x.split('_')[-1].split('.jpg')[0]))

        if not jpg_files:
            print("No jpg files found with the given prefix.")
            return

        img = cv2.imread(jpg_files[0])
        if img is None:
            print(f"Error reading the image: {jpg_files[0]}")
            return

        height, width, layers = img.shape

        combinations = [('XVID', 'avi')]

        for codec, ext in combinations:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            output_filename = f"{base_output_filename}_{codec}.{ext}"
            out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

            for file in jpg_files:
                img = cv2.imread(file)
                if img is not None:
                    out.write(img)
                else:
                    print(f"Error reading the image: {file}")

            out.release()
            print(f"Saved video with {codec} codec to {output_filename}")


if __name__ == '__main__':

    start_command = input("Press 's' to start: ")

    if start_command == 's':
        drone = Drone()
        yaw = 0
        pitch = 0
        step = 0
        drone.set_gimbal_angle(0, -90)
        time.sleep(1.5)
        drone.set_gimbal_angle(0, 0)
        time.sleep(1.5)


        def yaw_pitch(x, y, current_yaw, current_pitch, threshold=50, movement=2):
            x_conversion = x - 425
            y_conversion = y - 240
            if x_conversion > threshold:
                yaw_change = -movement
            elif x_conversion < -threshold:
                yaw_change = movement
            else:
                yaw_change = 0

            if y_conversion > threshold:
                pitch_change = movement
            elif y_conversion < -threshold:
                pitch_change = -movement
            else:
                pitch_change = 0

            if (current_yaw + yaw_change > 135) or (current_yaw + yaw_change < -135):
                yaw_change = 0
            if (current_pitch + pitch_change > 0) or (current_pitch + pitch_change) < -90:
                pitch_change = 0

            return yaw_change, pitch_change


        try:
            while True:
                step += 1
                sending_array = drone.detect_and_find_center()
                truth = 0
                if sending_array[1] != 240:
                    truth = 1

                sending_data = [sending_array[0], sending_array[1], truth]
                print(sending_data)

                drone.sending_data(sending_data)
                time.sleep(0.1)

                if step % 2 == 1:
                    yaw_change, pitch_change = yaw_pitch(sending_array[0], sending_array[1], yaw, pitch)
                    yaw += yaw_change
                    pitch += pitch_change
                    drone.set_gimbal_angle(yaw, pitch)

        except KeyboardInterrupt:
            drone.images_to_avi("captured_image", "output.avi")
            print("Video saved as output.avi")


