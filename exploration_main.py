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
        
        # server data send & receive
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.addr = None

        if not self.camera.isOpened():
            print("Error: Couldn't open the camera.")
            return
        
        #detection requirements
        # self.model = YOLO('Tech_piece/Detection/best3.onnx')
        self.CONFIDENCE_THRESHOLD = 0.1
        self.tracker = None
        self.success = False
        self.maxtrack = 180
        self.tframe = 0
        self.prevx = []
        self.prevy = []
        self.label = None
        self.labels = ['fixed', 'quadcopter', 'hybrid', 'label']

    # color camera test1
    def detect_and_find_center(self, x=1.3275, save_image=True, image_name="captured_image.jpg"):
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

        # Always draw the circle at the detected center (or default if no center detected)
        cv2.circle(res_frame, center, 10, (100, 100, 100), -1)

        if save_image:
            self.image_count += 1
            image_name = f"captured_image_{self.image_count}.jpg"
            cv2.imwrite(image_name, res_frame)

        return (center[0], 480 - center[1])

    #drone detection return [x, y, label] None if not detected
    def detect(self, x=1.3275, save_image=True, image_name="captured_image.jpg"):
        ret, frame = self.camera.read()
        conf = 0

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * x), h))

        #cam check
        if not ret:
            print('Cam Error')
            return None

        #Detection
        if (self.tracker is None) or (self.tframe > self.maxtrack):
            #detection = get_prediction(frame, self.detection_model)
            detection = self.model(frame, verbose=False)[0]
            #Sliced inference
            #detection = get_sliced_prediction(frame, self.detection_model, slice_height=480, slice_width=480, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
            for data in detection.boxes.data.tolist():
                confidence = float(data[4])

                print(len(detection.boxes.data.tolist()))
                print(confidence, self.CONFIDENCE_THRESHOLD)

                if (confidence > self.CONFIDENCE_THRESHOLD) and ((int(data[2]) - int(data[0])) < 500) and ((int(data[3]) - int(data[1])) < 500) and (confidence > conf):
                    xmin, ymin, xlen, ylen = int(data[0]), int(data[1]), int(data[2]) - int(data[0]), int(data[3]) - int(data[1])
                    xmid = xmin+xlen/2
                    ymid = ymin+ylen/2
                    conf = confidence
                    self.label = self.labels[int(data[5])]
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
                print(e)
                self.tracker = None
                pass

        #tracking
        try:
            self.success, roi = self.tracker.update(frame)
            self.tframe += 1
            # if self.success:  
            if True:
                (x, y, w, h) = tuple(map(int, roi))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if (x+w/2 < 5) or (x+w/2 > self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT) - 5) or (y+h/2 < 5) or (y+h/2 > self.camera.get(cv2.CAP_PROP_FRAME_WIDTH) - 5):
                    print('out of frame')
                    self.tracker = None
                loc = [x+w/2, y+h/2, self.label]

            if save_image:
                self.image_count += 1
                image_name = f"captured_image_{self.image_count}.jpg"
                cv2.imwrite(image_name, frame)

                return loc
            else:
                self.tracker = None
        except Exception as e:
            print(e)
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
    
    # altitude control for bounding box
    def altitude_ctr(self, length):
        if length < 100:
            return 1 # 올라가라
        elif length > 500:
            return 2 # 내려가라
        else:
            return 0 # 유지해라
        
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
            data_06 = response[index_0d+1:index_0d+7]
            yaw_raw, pitch_raw, roll_raw = struct.unpack('<hhh', data_06)

            # Yaw Velocity, Pitch Velocity, Roll Velocity 데이터를 추출합니다.
            data_0c = response[index_0d+7:index_0d+15]
            # print(len(data_0c))
            if len(data_0c) != 8:
                raise ValueError("Invalid data length for yaw_velocity_raw, pitch_velocity_raw, roll_velocity_raw")

            yaw_velocity_raw, pitch_velocity_raw, roll_velocity_raw, _ = struct.unpack('<hhhh', data_0c)

            # 추출한 데이터를 10으로 나눠 실제 값으로 변환합니다.
            yaw = yaw_raw / 10.0
            pitch = pitch_raw / 10.0
            if pitch < 0 :
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

    def angle_cali(y, pitch, standard_pitch = 0): # 기준 yaw = 0, pitch = -90 ### pitch = -60을 기준으로 하려면 숫자 90 -> 60 수정해야 함.
        y_new =  y + ((pitch - standard_pitch) * (130/15)) # 15도당 130프레임

        return y_new

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
   
    # modified 11/02
    def yaw_pitch(self, y, current_pitch, threshold=100, movement=4):
        y_conversion = y - 240

        if y_conversion > threshold:
            pitch_change = movement
        elif y_conversion < -threshold:
            pitch_change = -movement
        else:
            pitch_change = 0

        if (current_pitch + pitch_change < 0) or (current_pitch + pitch_change) > 90:
            pitch_change = 0
        # print("pitch:", pitch_change)
        return pitch_change
    
    # server 1
    def setup_connection(self):
        self.server_socket.bind(('192.168.0.31', 12345))
        self.server_socket.listen(5)
        print("Waiting for connection...")
        self.conn, self.addr = self.server_socket.accept()
        print("Connected by", self.addr)

    # server 2
    def receive_data(self):
        data = self.conn.recv(1024)
        decoded_data = data.decode('utf-8')
        print('Received message:', decoded_data)
        return decoded_data
    
    # server 3
    def send_data(self, data):
        try:
            self.conn.sendall(data.encode('utf-8'))
        except ConnectionResetError:
            print("Connection was reset by peer.")
            print("Attempting to reconnect...")
            
            # 연결을 닫고 다시 시작합니다.
            self.conn.close()
            self.setup_connection()
            
            # 재연결 후 다시 데이터를 전송해볼 수 있습니다.
            try:
                self.conn.sendall(data.encode('utf-8'))
            except Exception as e:
                print("Failed to send data after reconnection:", e)
    # server 4
    def close_connection(self):
        if self.conn:
            self.conn.close()
        self.server_socket.close()
        

### 초기 탐색 코드
# 카메라 pitch, 드론 yaw를 조절해가며 탐색한다.
# 카메라 pitch는 10 ~ 60 반복 (o)
# 드론 yaw는 -45 ~ 45 반복
# 탐지 성공 시 yaw 서쪽으로 고정하는 기동 수행

if __name__ == '__main__':

    start_command = input("Press 's' to start: ")

    if start_command == 's':

        drone = Drone()
        
        # drone.setup_connection() 
        # received_data = drone.receive_data()

        yaw = 0
        pitch = 0  # -45, -90
        step = 0
        drone.set_gimbal_angle(yaw, pitch) # 초기 각도
        time.sleep(1.5)
        
        direction = 5 
        cnt = 1
        try:
            while True:
                drone.set_gimbal_angle(0, pitch)
                print(pitch)
                if pitch == 60 or pitch == 10 and direction == -5:
                    direction = -direction
                pitch += direction
                time.sleep(0.1) 

                # if 인지 성공 시 종료 및 드론 기동
                sending_array = drone.detect_and_find_center()
                if sending_array[1] != 240:
                    cnt +=1
                elif cnt == 10:
                    break


        except KeyboardInterrupt:
            drone.images_to_avi("captured_image", "output.avi")
            print("Video saved as output.avi")
            drone.close_connection()
