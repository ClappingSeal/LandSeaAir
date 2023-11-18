from dronekit import connect
import cv2
import time
import socket
import struct
import logging
import numpy as np
import os
import math
import json
from ultralytics import YOLO
from datetime import datetime

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

        # Camera connection
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Error: Couldn't open the camera.")
            return

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

        # detection requirements
        self.model = YOLO('Tech_piece/Detection/best3.onnx')
        self.tracking_active = False
        self.scale_factor = 1
        self.tracker = None
        self.detection_in_detect2_for_detect3 = (425, 240, 0, 0, -0.7)
        self.detect_call_counter = 0
        self.detect2_threshold = 0.1
        self.rescheduled_count = 100

        self.frame_width = 640
        self.frame_height = 480
        self.frame_width_divide_2 = self.frame_width // 2
        self.frame_height_divide_2 = self.frame_height // 2

    def detect(self, img_piece):
        original_height, original_width = img_piece.shape[:2]
        new_width = int(original_width * 1.3275)
        img_piece = cv2.resize(img_piece, (new_width, original_height), interpolation=cv2.INTER_LINEAR)

        model = self.model

        self.detect_call_counter += 1

        def detect1(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centers = []
            sizes = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    centers.append((x + w // 2, y + h // 2))
                    sizes.append((w, h))

            if centers:
                avg_x = sum([c[0] for c in centers]) // len(centers)
                avg_y = sum([c[1] for c in centers]) // len(centers)

                min_size = max(sizes, key=lambda size: size[0] * size[1])
                width, height = min_size

                return avg_x, avg_y, width, height, -1

            return self.frame_width_divide_2, self.frame_height_divide_2, 0, 0, -1

        def detect2(img):
            CONFIDENCE_THRESHOLD = self.detect2_threshold
            detection = model(img, verbose=False)[0]

            highest_confidence = 0
            best_bbox = None

            for data in detection.boxes.data.tolist():
                confidence = float(data[4])
                if confidence > highest_confidence and confidence > CONFIDENCE_THRESHOLD:
                    highest_confidence = confidence
                    x_min = int(data[0])
                    y_min = int(data[1])
                    x_len = int(data[2]) - int(data[0])
                    y_len = int(data[3]) - int(data[1])
                    label_idx = int(data[5])

                    best_bbox = (x_min, y_min, x_len, y_len, label_idx)

            if best_bbox:
                self.detection_in_detect2_for_detect3 = best_bbox
                return best_bbox

            # 만약 조건을 만족하는 bounding box가 없다면 기본 값을 반환
            return self.frame_width_divide_2, self.frame_height_divide_2, 0, 0, -2

        def detect3(img):
            if self.detection_in_detect2_for_detect3:
                X, Y, width, height, label_idx = self.detection_in_detect2_for_detect3
                if width <= 0 or height <= 0 or X + width > img.shape[1] or Y + height > img.shape[0]:
                    return self.frame_width_divide_2, self.frame_height_divide_2, 0, 0, -3

                bbox = (X, Y, width, height)
                if self.tracker is None:
                    self.tracker = cv2.TrackerKCF_create()
                    self.tracker.init(img, bbox)

                self.success, bbox = self.tracker.update(img)
                X, Y, width, height = tuple(map(int, bbox))
                return X, Y, width, height, label_idx
            else:
                return self.frame_width_divide_2, self.frame_height_divide_2, 0, 0, -3

        x, y, w, h, label_idx = self.detection_in_detect2_for_detect3

        if self.detect_call_counter % self.rescheduled_count == 0:
            x, y, w, h, label_idx = detect2(img_piece)

        return detect2(img_piece)

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

    # gimbal 2 modified 11/02
    def set_gimbal_angle(self, yaw, pitch):  # 각도 체크섬 생성 및 각도 조종 명령 주기
        cmd_header = b'\x55\x66\x01\x04\x00\x00\x00\x0E'
        yaw_bytes = struct.pack('<h', int(yaw * 10))
        pitch_bytes = struct.pack('<h', int(-pitch * 10))
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

    # gimbal 6
    def accquire_data(self):
        self.send_command_to_gimbal(b'\x55\x66\x01\x00\x00\x00\x00\x0d\xe8\x05')

        try:
            response, addr = self.udp_socket.recvfrom(1024)
            # print("Received:", response)
            return response
        except socket.error as e:
            print(f"Error receiving data via UDP: {e}")
            return None

    # gimbal 7 modified 11/02
    def acquire_attitude(self, response):
        try:
            # CMD ID를 찾습니다.
            index_0d = response.find(b'\x0D')

            # Yaw, Pitch, Roll 데이터를 추출합니다.
            data_06 = response[index_0d + 1:index_0d + 7]
            yaw_raw, pitch_raw, roll_raw = struct.unpack('<hhh', data_06)

            # Yaw Velocity, Pitch Velocity, Roll Velocity 데이터를 추출합니다.
            data_0c = response[index_0d + 7:index_0d + 15]
            # print(len(data_0c))
            if len(data_0c) != 8:
                raise ValueError("Invalid data length for yaw_velocity_raw, pitch_velocity_raw, roll_velocity_raw")

            yaw_velocity_raw, pitch_velocity_raw, roll_velocity_raw, _ = struct.unpack('<hhhh', data_0c)

            # 추출한 데이터를 10으로 나눠 실제 값으로 변환합니다.
            yaw = yaw_raw / 10.0
            pitch = pitch_raw / 10.0
            if pitch < 0:
                pitch = -(180 + pitch)
            else:
                pitch = 180 - pitch
            roll = roll_raw / 10.0
            yaw_velocity = yaw_velocity_raw / 10.0
            pitch_velocity = pitch_velocity_raw / 10.0
            roll_velocity = roll_velocity_raw / 10.0

            return yaw, pitch, roll, yaw_velocity, pitch_velocity, roll_velocity
        except struct.error as e:
            print("Error in unpacking data: {}".format(e))
            return 10000, 10000, 10000, 10000, 10000, 10000
        except ValueError as e:
            print("Error: {}".format(e))
            return 10000, 10000, 10000, 10000, 10000, 10000

    # gimbal 8 added 11/04
    def set_gimbal_angle_feedback(self, yaw, pitch):
        for i in range(5):
            self.set_gimbal_angle(yaw, pitch)
            yaw_set = self.current_yaw
            pitch_set = self.current_pitch

            response = self.accquire_data()
            yaw_current, pitch_current, _, _, _, _ = self.acquire_attitude(response)
            if abs(yaw_set - yaw_current) < 5 and abs(pitch_set - pitch_current) < 5:
                print("all set angles")
                break
            time.sleep(2)

    # end
    def close_connection(self):
        self.vehicle.close()

    # to avi
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
        drone.set_gimbal_angle(0, 30)
        time.sleep(2)

        image_counter = 1

        try:
            while True:
                ret, frame = drone.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                sending_array = drone.detect(frame)
                sending_data = [sending_array[0], sending_array[1], sending_array[2], sending_array[3]]

                # sending data
                drone.sending_data(sending_data)

                # image show, process
                x, y, w, h, label_idx = sending_array
                x_center = x + w // 2
                y_center = drone.frame_height - (y + h // 2)
                print(x_center, y_center, w, h, label_idx)

                # Draw bounding box
                if w > 0 and h > 0:  # if width and height are positive, we have a valid box to draw
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f'Label {label_idx}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Display the resulting frame
                # cv2.imshow('Drone Camera Feed', frame)

                 current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 이미지에 시간 표시
                cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
  
                filename = f"{image_counter}.jpg"
                cv2.imwrite(filename, frame)
                image_counter += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            drone.close_connection()
            drone.camera.release()
            drone.images_to_avi("", "output.avi")
            cv2.destroyAllWindows()
            
            

