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
        self.base_color = np.array([0, 255, 255])
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

        # detection requirements
        self.model = YOLO('Tech_piece/Detection/best3.onnx')
        self.confidence_threshold = 0.75
        self.scale_factor = 1.3275
        self.capture_count = 0
        self.label = None
        self.labels = ['fixed', 'quadcopter', 'hybrid']
        self.previous_centers = []
        self.center_count = 3
        self.tolerance = 100
        self.tracker_initialized = False
        self.tracker = None
        self.frame_count = 0
        self.recheck_interval = 10  # 드론 재확인 간격

    # color camera test1
    def detect_and_find_center(self, x=1.3275, save_image=True):
        ret, frame = self.camera.read()
        if not ret:
            print("Error: Couldn't read frame.")
            return (425, 240)

        # Resize frame considering the aspect ratio multiplier
        h, w = frame.shape[:2]
        res_frame = cv2.resize(frame, (int(w * x), h))

        hsv = cv2.cvtColor(res_frame, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([self.base_color[0] - self.threshold, 130, 130])
        upper_bound = np.array([self.base_color[0] + self.threshold, 255, 255])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center = (425, 240)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY)
                print('sadf')

        # Always draw the circle at the detected center (or default if no center detected)
        cv2.circle(res_frame, center, 10, (100, 100, 100), -1)

        if save_image:
            self.image_count += 1
            image_name = f"captured_image_{self.image_count}.jpg"
            cv2.imwrite(image_name, res_frame)

        return (center[0], 480 - center[1])

    # drone camera 1 (drone detection return [x, y, label] None if not detected)
    def __del__(self):
        self.camera.release()

    # drone camera 2
    def detect(self):
        ret, frame = self.camera.read()
        if not ret:
            return 425, 240, 0, 0

        frame_resized = cv2.resize(frame, None, fx=self.scale_factor, fy=1)
        self.frame_count += 1

        if self.tracker_initialized:
            success, bbox = self.tracker.update(frame_resized)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 추적된 객체를 빨간색으로 표시

                if self.frame_count % self.recheck_interval == 0:
                    if not self.is_drone(frame_resized, bbox):
                        self.tracker_initialized = False  # 드론이 아니라면 트래커 초기화
                # 추적하는 동안 이미지 저장
                cv2.imwrite(f"captured_image_{self.capture_count}.jpg", frame_resized)
                self.capture_count += 1
                return x + w // 2, 480 - (y + h // 2), w, h  # 중심 좌표 반환
            else:
                self.tracker_initialized = False  # 추적 실패 시 초기화

        detection = self.model(frame_resized, verbose=False)[0]
        best_confidence = 0
        best_data = None
        for data in detection.boxes.data.tolist():
            confidence = float(data[4])
            if confidence > best_confidence:
                best_confidence = confidence
                best_data = data

        if best_data and best_confidence > self.confidence_threshold:
            center_x, center_y, width, height = self.get_center_and_dimensions(best_data)
            cv2.rectangle(frame_resized, (center_x - width // 2, center_y - height // 2),
                          (center_x + width // 2, center_y + height // 2), (0, 255, 0), 2)
            cv2.imwrite(f"captured_image_{self.capture_count}.jpg", frame_resized)
            self.capture_count += 1

            self.previous_centers.append((center_x, center_y))
            if len(self.previous_centers) > self.center_count:
                self.previous_centers.pop(0)

            if not self.tracker_initialized:
                bbox = (center_x - width // 2, center_y - height // 2, width, height)
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(frame_resized, bbox)
                self.tracker_initialized = True

        cv2.imwrite(f"captured_image_{self.capture_count}.jpg", frame_resized)
        self.capture_count += 1

        if best_data and best_confidence > self.confidence_threshold:
            return center_x, 480 - center_y, width, height
        else:
            return 425, 240, 0, 0

    # drone camera 3
    def is_drone(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        cropped_frame = frame[y:y + h, x:x + w]

        detection = self.model(cropped_frame, verbose=False)[0]
        for data in detection.boxes.data.tolist():
            confidence = float(data[4])
            if confidence > self.confidence_threshold:
                return True
        return False

    # drone camera 4
    @staticmethod
    def get_center_and_dimensions(data):
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
        width = xmax - xmin
        height = ymax - ymin
        return center_x, center_y, width, height

    # drone camera 5
    def check_stability(self):
        if len(self.previous_centers) < self.center_count:
            return False

        distances = []
        for i in range(1, len(self.previous_centers)):
            prev_center = self.previous_centers[i - 1]
            curr_center = self.previous_centers[i]
            distance = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)
            distances.append(distance)

        return all(abs(d - distances[0]) < self.tolerance for d in distances)

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
    def yaw_pitch(self, x, y, current_yaw, current_pitch, threshold=50, movement=2):
        x_conversion = x - 425
        y_conversion = y - 240
        if x_conversion > threshold:
            yaw_change = -movement
        elif x_conversion < -threshold:
            yaw_change = movement
        else:
            yaw_change = 0

        if y_conversion > threshold / 2:
            pitch_change = movement
        elif y_conversion < -threshold / 2:
            pitch_change = -movement
        else:
            pitch_change = 0

        if (current_yaw + yaw_change > 135) or (current_yaw + yaw_change < -135):
            yaw_change = 0
        if (current_pitch + pitch_change > 0) or (current_pitch + pitch_change) < -90:
            pitch_change = 0

        return yaw_change, pitch_change

    # gimbal 5
    def accquire_data(self):
        self.send_command_to_gimbal(b'\x55\x66\x01\x00\x00\x00\x00\x0d\xe8\x05')
        
        try:
            response, addr = self.udp_socket.recvfrom(1024) 
            # print("Received:", response)
            return response
        except socket.error as e:
            print(f"Error receiving data via UDP: {e}")
            return None
    
    # gimbal 6
    def acquire_attitude(self,response):
        # CMD ID를 찾습니다.
        index_0d = response.find(b'\x0D')

        # Yaw, Pitch, Roll 데이터를 추출합니다.
        data_06 = response[index_0d+1:index_0d+7]
        yaw_raw, pitch_raw, roll_raw = struct.unpack('<hhh', data_06)

        # Yaw Velocity, Pitch Velocity, Roll Velocity 데이터를 추출합니다.
        data_0c = response[index_0d+7:index_0d+15]
        yaw_velocity_raw, pitch_velocity_raw, roll_velocity_raw, _ = struct.unpack('<hhhh', data_0c)

        # 추출한 데이터를 10으로 나눠 실제 값으로 변환합니다.
        yaw = yaw_raw / 10.0
        pitch = pitch_raw / 10.0
        roll = roll_raw / 10.0
        yaw_velocity = yaw_velocity_raw / 10.0
        pitch_velocity = pitch_velocity_raw / 10.0
        roll_velocity = roll_velocity_raw / 10.0

        return yaw, pitch, roll, yaw_velocity, pitch_velocity, roll_velocity
    
    def capture_image(self, num1, num2, num3, num4, num5):
        ret, frame = self.camera.read()

        if ret:
            file_name = f"{num1}_and_{num2}_vs_{num3}_and_{num4}_and_{num5}.jpg"
            cv2.imwrite(file_name, frame)
            print(f"Picture saved as {file_name}.")
        else:
            print("Cannot take picture.")

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
        drone.capture_image(0,0,0,0)
        drone.capture_image(0,0,0,0)
        drone.capture_image(0,0,0,0)
        drone.capture_image(0,0,0,0)
        # yaws = [90, 75, 60, 45, 30, 15, 0, -15, -30, -45, -60, -75, -90]
        yaws = [0,15,30,45,60,75,90,-15, -30, -45, -60, -75, -90]
        pitches = [0, -15, -30, -45, -60, -75, -90]

        for yaw in yaws:
            for pitch in pitches:
                drone.set_gimbal_angle(yaw, pitch)
                time.sleep(3)
                while True:
                    response = drone.accquire_data()
                    yaw_wr, pitch_wr, roll_wr, _, _, _ = drone.acquire_attitude(response)
                    if abs(yaw_wr - yaw) < 5 and abs(pitch_wr - pitch) < 5:
                        print("Yaw:", yaw_wr)
                        print("Pitch:", pitch_wr)
                        print("Roll:", roll_wr)
                        break
                time.sleep(0.2)
                drone.capture_image(yaw, pitch, yaw_wr, pitch_wr, roll_wr)
                time.sleep(0.2)

        try:
            while True:
                print('finish!!!')

                time.sleep(0.1)

        except KeyboardInterrupt:
            drone.images_to_avi("captured_image", "output.avi")
            print("Video saved as output.avi")
            drone.close_connection()

