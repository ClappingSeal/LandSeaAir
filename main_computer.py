from dronekit import connect, VehicleMode, Command, LocationGlobalRelative
from pymavlink import mavutil
import time
import logging
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class Drone:
    def __init__(self, connection_string='COM14', baudrate=57600):
        print('vehicle connecting...')

        # Connecting value
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.vehicle = connect(self.connection_string, wait_ready=True, baud=self.baudrate, timeout=100)
        # self.vehicle = connect('tcp:127.0.0.1:5762', wait_ready=False, timeout=100)

        # Communication
        self.standard_pit = 70
        self.received_data = (425, 240, 0, 0, 0, self.standard_pit, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.vehicle.add_message_listener('DATA64', self.data64_callback)

        # DRL model load
        self.model = PPO.load("ppo_model")

        # Position value
        self.init_lat = self.vehicle.location.global_relative_frame.lat
        self.init_lon = self.vehicle.location.global_relative_frame.lon

        # Arming value
        self.min_throttle = 1000
        self.arm_throttle = 1200
        #
        # # Check position
        # if self.init_lat is None or self.init_lon is None:
        #     raise ValueError("Latitude or Longitude value is None. Class initialization aborted.")
        # print("Drone current location : ", self.init_lat, "lat, ", self.init_lon, "lon")
        #
        # if self.init_lat == 0 or self.init_lon == 0:
        #     raise ValueError("Cannot get Location. Class initialization aborted.")

        # pid d value
        self.past_pos_data = np.zeros((30, 2))

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
        self.vehicle.channels.overrides['3'] = self.min_throttle
        self.vehicle.mode = VehicleMode("STABILIZE")
        time.sleep(0.5)

        cmds = self.vehicle.commands
        cmds.download()
        cmds.wait_ready()
        cmds.clear()
        takeoff_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                              mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, h)
        cmds.add(takeoff_cmd)
        cmds.upload()
        time.sleep(0.5)  # upload wait

        self.vehicle.armed = True
        time.sleep(0.5)
        self.vehicle.channels.overrides['3'] = self.arm_throttle
        time.sleep(3)
        print("ARMED : ", self.vehicle.armed)

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
    def set_yaw_to_west(self):
        yaw_angle = 270
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

        print("Setting yaw to face WEST!!!!!!!!!!!!!!!!!!!!")
        time.sleep(0.5)

    # Drone movement3 non-block
    def set_yaw_to_west_nonblock(self):
        yaw_angle = 270
        is_relative = False

        self.vehicle._master.mav.command_long_send(
            self.vehicle._master.target_system, self.vehicle._master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0,
            yaw_angle, 0, 0, is_relative, 0, 0, 0)

        self.target_yaw = yaw_angle
        self.yaw_tolerance = 0.1  # 단위는 도

    # Drone movement4 non-block
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
            vx, -vy, -vz,  # x, y, z 속도
            0, 0, 0,  # x, y, z 가속도
            0, 0)  # yaw, yaw_rate
        self.vehicle.send_mavlink(msg)

    # Drone movement5 non-block (velocity 함수 사용)
    def velocity_pid(self, target_x, target_y, velocity_z, history_positions, proportional=0.6, integral=0.001,
                     derivative=0.5):
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
        self.velocity(velocity_x, velocity_y, velocity_z)

    # Drone movement6 non-block
    def goto_location(self, x, y, z, speed=10):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        target_lat = self.init_lat + (x / LATITUDE_CONVERSION)
        target_lon = self.init_lon - (y / LONGITUDE_CONVERSION)
        target_alt = z

        if self.vehicle.mode != VehicleMode("GUIDED"):
            self.vehicle.mode = VehicleMode("GUIDED")
            time.sleep(0.1)

        target_location = LocationGlobalRelative(target_lat, target_lon, target_alt)

        self.vehicle.groundspeed = speed
        self.vehicle.simple_goto(target_location)

        # print(f"Moving to: Lat: {target_lat}, Lon: {target_lon}, Alt: {target_alt} at {speed} m/s")

    # Drone movement7 block (get_pos 함수 사용)
    def goto_location_block(self, x, y, z):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        print(self.init_lat, y, LONGITUDE_CONVERSION)

        target_lat = self.init_lat + (x / LATITUDE_CONVERSION)
        target_lon = self.init_lon - (y / LONGITUDE_CONVERSION)
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

    # Drone movement8 block
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

    # locking 1 (get_pos 함수, velocity_pid 함수 사용)
    def locking_easy(self, x, y, num):
        x_conversion = (x - 425) / num
        y_conversion = (y - 240) / num
        target_x = self.get_pos()[0] + x_conversion
        target_y = self.get_pos()[1] + y_conversion
        self.velocity_pid(target_x, target_y, self.past_pos_data)
        print(target_x, target_y)

    # locking 2
    def locking_drl(self, yaw_cam, pitch_cam, x_frame, y_frame, vel_z=0):
        # 카메라 초기 각도
        standard_pitch = self.standard_pit
        standard_yaw = 0
        # 프레임 변경
        x_frame = x_frame + ((yaw_cam - standard_yaw) * (130 / 15))
        y_frame = y_frame + ((pitch_cam - standard_pitch) * (130 / 15))

        obs = np.array([(x_frame - 425) / 10, (y_frame - 240) / 10])
        action, _ = self.model.predict(obs)

        speed_magnitude = np.linalg.norm(action)
        if speed_magnitude > 10:
            action = (action / speed_magnitude) * 10

        x_conversion = -action[0] * 0.5  # Scale
        y_conversion = action[1] * 0.5  # Scale
        target_x = self.get_pos()[0] + x_conversion
        target_y = self.get_pos()[1] + y_conversion
        self.velocity_pid(target_x, target_y, vel_z, self.past_pos_data)
        print(x_conversion, "and", y_conversion)

    # get position (m)
    def get_pos(self):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        delta_lat = self.vehicle.location.global_relative_frame.lat - self.init_lat
        delta_lon = self.vehicle.location.global_relative_frame.lon - self.init_lon

        x = delta_lat * LATITUDE_CONVERSION
        y = -delta_lon * LONGITUDE_CONVERSION

        return x, y

    # update past data position by rolling
    def update_past_pos_data(self):
        self.past_pos_data = np.roll(self.past_pos_data, shift=-1, axis=0)
        self.past_pos_data[-1] = self.get_pos()

    # IMM target tracking (4x2 행렬만 처리, 30x2 행렬만 입력)
    def imm_tracking(self, data_raw, num_steps=2):
        def compute_velocity_and_acceleration(values):
            velocities = np.diff(values) / np.diff(times)
            accelerations = np.diff(velocities)
            return velocities[-1], accelerations[-1]

        def predict_next_step(value, velocity, acceleration, delta_t):
            value_cv = value + velocity * delta_t
            value_ca = value + velocity * delta_t + 0.5 * acceleration * delta_t ** 2
            w1, w2 = 0.6, 0.4  # 모델 가중치
            return w1 * value_cv + w2 * value_ca

        def select_rows(matrix):
            # 행렬의 1, 10, 20, 30번째 행을 선택
            selected_rows = matrix[[0, 9, 19, 29], :]
            return selected_rows

        data = select_rows(data_raw)

        data = np.array(data)
        times = np.arange(1, data.shape[0] + 1)
        x_values = data[:, 0]
        y_values = data[:, 1]

        vx, ax = compute_velocity_and_acceleration(x_values)
        vy, ay = compute_velocity_and_acceleration(y_values)

        predictions = []
        delta_t_pred = 1  # 예측에 사용할 시간 간격

        x_next, y_next = x_values[-1], y_values[-1]

        for _ in range(num_steps):
            x_next = predict_next_step(x_next, vx, ax, delta_t_pred)
            y_next = predict_next_step(y_next, vy, ay, delta_t_pred)

            vx += ax * delta_t_pred
            vy += ay * delta_t_pred

            predictions.append((x_next, y_next))

        return predictions

    # get battery
    def battery_state(self):
        return self.vehicle.battery.voltage

    # end
    def close_connection(self):
        self.vehicle.close()


if __name__ == "__main__":
    gt = Drone()
    difference = 100

    try:
        nums = 1, 1

        if len(nums) == 2:
            # gt.arm_takeoff(1)
            # gt.set_yaw_to_west()

            time.sleep(0.1)

            step = 0
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_xlim(-3000, 3000)  # x축 범위 설정
            ax.set_ylim(-3000, 3000)  # y축 범위 설정

            while True:
                step += 1
                receive_arr = np.array(gt.receiving_data())
                # print(receive_arr)

                gt.update_past_pos_data()
                gt.set_yaw_to_west_nonblock()

                # print(gt.get_pos())

                if step % 30 == 0:
                    current_pos = gt.get_pos()
                    print(current_pos)
                    ax.scatter(-current_pos[1], current_pos[0])
                    plt.draw()
                    plt.pause(0.1)
                    difference = difference * -1

                time.sleep(0.1)
                print(receive_arr[4], receive_arr[5], receive_arr[0], receive_arr[1])

                gt.locking_drl(receive_arr[4], receive_arr[5], receive_arr[0], receive_arr[1])

        else:
            print("정확하게 두 개의 실수를 입력하세요.")

    except ValueError:
        print("올바른 형식의 실수를 입력하세요.")
    except KeyboardInterrupt:
        gt.land()
        gt.close_connection()
