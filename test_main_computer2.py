from dronekit import connect, VehicleMode, Command, LocationGlobalRelative
from pymavlink import mavutil
import time
import logging
import math
import numpy as np
from stable_baselines3 import PPO
import paramiko
import socket
import json

logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class Drone:
    def __init__(self, connection_string='COM14', baudrate=57600):
        print('vehicle connecting...')

        # Connecting value
        self.connection_string = connection_string
        self.baudrate = baudrate
        # self.vehicle = connect(self.connection_string, wait_ready=False, baud=self.baudrate, timeout=100)
        self.vehicle = connect('tcp:127.0.0.1:5762', wait_ready=False, timeout=100)

        # Communication
        self.received_data = (425, 240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.vehicle.add_message_listener('DATA64', self.data64_callback)

        # DRL model load
        self.model = PPO.load("ppo_model")

        # Position value
        self.init_lat = self.vehicle.location.global_relative_frame.lat
        self.init_lon = self.vehicle.location.global_relative_frame.lon
        if self.init_lat is None or self.init_lon is None:
            raise ValueError("Latitude or Longitude value is None. Class initialization aborted.")
        print(self.init_lat, self.init_lon)

        self.past_pos_data = np.zeros((20, 2))
        
        # client setup
        self.client_socket = None

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

    # Drone movement1 block
    def arm_takeoff(self, h):

        self.vehicle.mode = VehicleMode("STABILIZE")
        cmds = self.vehicle.commands
        cmds.download()
        cmds.wait_ready()
        cmds.clear()
        takeoff_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                              mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, h)
        cmds.add(takeoff_cmd)
        cmds.upload()

        time.sleep(0.1)
        self.vehicle.armed = True
        time.sleep(0.1)
        while not self.vehicle.armed:
            print("Waiting for arming...")
            time.sleep(1)

        self.vehicle._master.mav.command_long_send(
            self.vehicle._master.target_system, self.vehicle._master.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START, 0,
            0, 0, 0, 0, 0, 0, 0, 0)
        time.sleep(2)

        print("Mission started")

        while True:
            print(f"Altitude: {self.vehicle.location.global_relative_frame.alt}")
            if self.vehicle.location.global_relative_frame.alt >= h * 0.8:
                print("Reached target altitude!!!!!!!!!!!!!!!!!!!!")
                break
            time.sleep(1)

        self.vehicle.mode = VehicleMode("GUIDED")

    # Drone movement2 block
    def set_yaw_to_north(self):
        yaw_angle = 0
        is_relative = False

        self.vehicle._master.mav.command_long_send(
            self.vehicle._master.target_system, self.vehicle._master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0,
            yaw_angle, 0, 0, is_relative, 0, 0, 0)

        tolerance = 5  # 단위는 도

        while True:
            current_yaw = self.vehicle.attitude.yaw
            current_yaw_deg = math.degrees(current_yaw) % 360
            angle = abs(current_yaw_deg - yaw_angle) % 360

            if min(angle, 360 - angle) <= tolerance:
                break
            print("Yaw : ", min(angle, 360 - angle))
            time.sleep(0.5)

        print("Setting yaw to face North!!!!!!!!!!!!!!!!!!!!")
        time.sleep(0.5)

    # Drone movement3 non-block
    def velocity(self, vx, vy, vz):
        if self.vehicle.mode != VehicleMode("GUIDED"):
            self.vehicle.mode = VehicleMode("GUIDED")
            time.sleep(0.1)

        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,  # boot time
            0, 0,  # target system, target component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # coordinate frame
            0b0000111111000111,  # type mask (enabling only velocity)
            0, 0, 0,  # x, y, z 위치
            vy, vx, vz,  # x, y, z 속도
            0, 0, 0,  # x, y, z 가속도
            0, 0)  # yaw, yaw_rate
        self.vehicle.send_mavlink(msg)

    # Drone movement4 non-block (velocity 함수 사용)
    def velocity_pid(self, target_x, target_y, history_positions, proportional=0.6, integral=0.001, derivative=0.5):
        pos_x, pos_y = self.get_pos()

        error_x = target_x - pos_x
        error_y = target_y - pos_y
        cumulative_error_x = sum([target_x - pos[0] for pos in history_positions])
        cumulative_error_y = sum([target_y - pos[1] for pos in history_positions])
        previous_error_x = target_x - history_positions[-10][0]
        previous_error_y = target_y - history_positions[-10][1]
        error_delta_x = error_x - previous_error_x
        error_delta_y = error_y - previous_error_y

        velocity_x = proportional * error_x + integral * cumulative_error_x + derivative * error_delta_x
        velocity_y = proportional * error_y + integral * cumulative_error_y + derivative * error_delta_y
        self.velocity(velocity_x, velocity_y, 0)

    # Drone movement5 non-block
    def goto_location(self, x, y, z, speed=10):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        target_lat = self.init_lat + (y / LATITUDE_CONVERSION)
        target_lon = self.init_lon + (x / LONGITUDE_CONVERSION)
        target_alt = z

        if self.vehicle.mode != VehicleMode("GUIDED"):
            self.vehicle.mode = VehicleMode("GUIDED")
            time.sleep(0.1)

        target_location = LocationGlobalRelative(target_lat, target_lon, target_alt)

        self.vehicle.groundspeed = speed
        self.vehicle.simple_goto(target_location)

        # print(f"Moving to: Lat: {target_lat}, Lon: {target_lon}, Alt: {target_alt} at {speed} m/s")

    # Drone movement6 block (get_pos 함수 사용)
    def goto_location_block(self, x, y, z):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        print(self.init_lat, y, LONGITUDE_CONVERSION)

        target_lat = self.init_lat + (y / LATITUDE_CONVERSION)
        target_lon = self.init_lon + (x / LONGITUDE_CONVERSION)
        target_alt = z

        def get_distance(lat1, lon1, lat2, lon2):
            import math
            R = 6371000  # Earth radius in meters

            d_lat = math.radians(lat2 - lat1)
            d_lon = math.radians(lon2 - lon1)

            a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(d_lon / 2) * math.sin(d_lon / 2))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            return R * c

        if self.vehicle.mode != VehicleMode("GUIDED"):
            self.vehicle.mode = VehicleMode("GUIDED")
            time.sleep(0.1)

        target_location = LocationGlobalRelative(target_lat, target_lon, target_alt)
        self.vehicle.simple_goto(target_location)
        print(f"Moving to: Lat: {target_lat}, Lon: {target_lon}, Alt: {target_alt}")

        while True:
            current_location = self.vehicle.location.global_relative_frame
            distance_to_target = get_distance(current_location.lat, current_location.lon, target_lat, target_lon)
            alt_diff = abs(current_location.alt - target_alt)
            print("current pos : ", self.get_pos())

            if distance_to_target < 1 and alt_diff < 1:
                print("Arrived at target location!!!!!!!!!!!!!!!!!!!!!")
                break
            time.sleep(0.5)

    # locking 1 (get_pos 함수, velocity_pid 함수 사용)
    def locking_easy(self, x, y, num):
        x_conversion = (x - 425) / num
        y_conversion = (y - 240) / num
        target_x = self.get_pos()[0] + x_conversion
        target_y = self.get_pos()[1] + y_conversion
        self.velocity_pid(target_x, target_y, self.past_pos_data)
        print(target_x, target_y)

    # Drone movement7 block
    def land(self):
        print("Initiating landing sequence")
        self.vehicle._master.mav.command_long_send(
            self.vehicle._master.target_system, self.vehicle._master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND, 0,
            0, 0, 0, 0, 0, 0, 0, 0)

        while self.vehicle.location.global_relative_frame.alt > 0.1:
            print(f"Altitude: {self.vehicle.location.global_relative_frame.alt}")
            time.sleep(1)
        print("Landed successfully!!!!!!!!!!!!!!!!!!!!")

    # get position (m)
    def get_pos(self):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        delta_lat = self.vehicle.location.global_relative_frame.lat - self.init_lat
        delta_lon = self.vehicle.location.global_relative_frame.lon - self.init_lon

        y = delta_lat * LATITUDE_CONVERSION
        x = delta_lon * LONGITUDE_CONVERSION

        return x, y

    # update past data position by rolling
    def update_past_pos_data(self):
        self.past_pos_data = np.roll(self.past_pos_data, shift=-1, axis=0)
        self.past_pos_data[-1] = self.get_pos()

    # get battery
    def battery_state(self):
        return self.vehicle.battery.voltage

    # end
    def close_connection(self):
        self.vehicle.close()


    def locking_drl(self, x_frame, y_frame):
        obs = np.array([x_frame, y_frame])
        action, _ = self.model.predict(obs)
        return -action

    # client 1
    def set_connection(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(("192.168.0.31",12345))
    # client 2
    def receive_data(self):
        data = self.client_socket.recv(1024).decode('utf-8')
        return data
    # client 3
    def send_data(self, data_to_send):
        self.client_socket.send(data_to_send.encode('utf-8'))
        reply = self.client_socket.recv(1024).decode('utf-8')
        return reply
    # client 4
    def close_connection(self):
        if self.client_socket:
            self.client_socket.close()

if __name__ == "__main__":
    gt = Drone()

    gt.set_connection() # client

    try:
        # raw_input = input("위도, 경도: ")

        nums = 1, 1
        # nums = [float(num.strip()) for num in raw_input.split(",")]

        # 미션 시작1
        if len(nums) == 2:
            gt.arm_takeoff(2)
            # gt.set_yaw_to_north()
            # time.sleep(0.1)

            while True:
                # client data receive
                data_received = gt.receive_data()
                data_list = json.loads(data_received)
                print("data_received")
                print(data_list)

                gt.sending_data([7, 80, 35, 8])
                receive_arr = np.array(gt.receiving_data())
                print(receive_arr)
                gt.locking_easy(receive_arr[0], receive_arr[1], 300) # 마지막 숫자가 줄어들면 빨라짐
                gt.update_past_pos_data()
                time.sleep(0.1)
                # print(gt.battery_state())

        else:
            print("정확하게 두 개의 실수를 입력하세요.")

    except ValueError:
        print("올바른 형식의 실수를 입력하세요.")
    except KeyboardInterrupt:
        gt.goto_location_block(0, 0, 5)
        gt.set_yaw_to_north()
        gt.land()
        gt.close_connection()
