from dronekit import Command, VehicleMode, connect
from pymavlink.dialects.v20 import ardupilotmega
import time
import logging
import threading

logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class Drone:
    def __init__(self, connection_string='COM5', baudrate=57600):
        print('vehicle connecting...')
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.vehicle = connect(self.connection_string, wait_ready=True, baud=self.baudrate)
        self.vehicle.add_message_listener('DATA64', self.on_data64)
        self.received_data = None
        self.data = None
        self.async_result = None

    # 통신 함수

    def on_data64(self, vehicle, name, message):
        if isinstance(message, ardupilotmega.MAVLink_data64_message):
            data = [int.from_bytes(message.data[i:i + 4], 'little') for i in range(0, len(message.data), 4)]
            self.received_data = data

    def get_received_data(self):
        return self.received_data

    def close_connection(self):
        self.vehicle.close()

    @staticmethod
    def asynchronous_received_data(drone_receiver):
        while True:
            if drone_receiver.get_received_data():
                a = drone_receiver.get_received_data()
                drone_receiver.async_result = a
                time.sleep(0.1)

    # 드론 액션 함수

    def set_flight_mode_by_pwm(self, pwm_value):
        def set_rc_channel_pwm(channel, pwm_value):
            if channel < 1:
                return
            self.vehicle.channels.overrides[channel] = pwm_value

        if 0 <= pwm_value <= 1230:
            mode = 'STABILIZE'
        elif 1231 <= pwm_value <= 1360:
            mode = 'AUTO'
        elif 1361 <= pwm_value <= 1490:
            mode = 'GUIDED'
        else:
            print("Another PWM value.")
            return

        set_rc_channel_pwm(5, pwm_value)
        print(f"mode to {mode}...")

    def arm(self):
        try:
            self.vehicle.channels.overrides['3'] = 1000
            self.set_flight_mode_by_pwm(1000)  # pwm signal for Stabilize mode
            time.sleep(3)
            self.vehicle.armed = True
            time.sleep(3)  # Wait for the drone to be armed
            if self.vehicle.armed:
                print("Vehicle armed")
            else:
                print("Vehicle armed fail")
        except APIException as e:
            print(str(e))

    def takeoff(self, h):
        self.set_flight_mode_by_pwm(1000)
        time.sleep(0.1)
        self.vehicle.channels.overrides['3'] = 1500
        time.sleep(2)

        cmds = self.vehicle.commands
        cmds.download()
        cmds.wait_ready()
        cmds.clear()
        takeoff_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                              mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, h)
        cmds.add(takeoff_cmd)
        cmds.upload()

        self.set_flight_mode_by_pwm(1300)  # pwm signal for AUTO mode
        time.sleep(0.1)

        while not self.vehicle.armed:
            print("Waiting for arming...")
            time.sleep(1)

        print("Taking off!")

        while True:
            print(f"Altitude: {self.vehicle.location.global_relative_frame.alt}")
            if self.vehicle.location.global_relative_frame.alt >= h * 0.8:
                print("Reached target altitude")
                break
            time.sleep(0.5)

    def goto(self, x, y, z):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        base_lat = 35.2265867
        base_lon = 126.8397070

        target_lat = base_lat + (y / LATITUDE_CONVERSION)
        target_lon = base_lon + (x / LONGITUDE_CONVERSION)

        self.set_flight_mode_by_pwm(1400)
        time.sleep(0.1)

        cmds = self.vehicle.commands
        cmds.download()
        cmds.wait_ready()
        cmds.clear()

        goto_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0, target_lat, target_lon, z)
        cmds.add(goto_cmd)
        cmds.upload()

        self.vehicle.mode = VehicleMode("AUTO")
        self.set_flight_mode_by_pwm(1300)  # pwm signal for AUTO mode

        # Monitoring the position until it reaches the target
        while math.dist([self.vehicle.location.global_relative_frame.lat,
                         self.vehicle.location.global_relative_frame.lon,
                         self.vehicle.location.global_relative_frame.alt],
                        [target_lat, target_lon, z]) > 0.5:  # Adjust as needed
            print(f"Current Position: {self.vehicle.location.global_relative_frame}")
            time.sleep(0.5)
        print("Reached target position")

    def land_by_auto_mode(self):
        self.set_flight_mode_by_pwm(1400)
        time.sleep(0.1)
        print("Landing using AUTO mode...")
        cmds = self.vehicle.commands
        cmds.download()
        cmds.wait_ready()
        cmds.clear()

        land_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0,
                           self.vehicle.location.global_relative_frame.lat,
                           self.vehicle.location.global_relative_frame.lon, 0)  # 0 for altitude as it's landing

        cmds.add(land_cmd)
        cmds.upload()

        self.vehicle.mode = VehicleMode("AUTO")
        self.set_flight_mode_by_pwm(1300)  # pwm signal for AUTO mode

        while self.vehicle.location.global_relative_frame.alt > 0.1:
            print(f"Altitude: {self.vehicle.location.global_relative_frame.alt}")
            time.sleep(1)

        print("Landed successfully!")


if __name__ == "__main__":
    # 드론 연결 및 데이터 수신 async 처리
    drone = Drone()
    data_thread = threading.Thread(target=Drone.asynchronous_received_data, args=(drone,))
    data_thread.start()

    # 드론 행동 블럭
    try:
        # 입력 (위도, 경도)
        raw_input = input("두 개의 실수를 입력하세요: ")
        nums = [float(num.strip()) for num in raw_input.split(",")]

        # 미션 시작
        if len(nums) == 2:
            while True:
                print(drone.async_result)
                time.sleep(0.1)

        else:
            print("정확하게 두 개의 실수를 입력하세요.")

    except ValueError:
        print("올바른 형식의 실수를 입력하세요.")
    except KeyboardInterrupt:
        drone.close_connection()


