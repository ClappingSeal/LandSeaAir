from dronekit import Command, VehicleMode, connect, LocationGlobalRelative, APIException
from pymavlink import mavutil
import time
import logging
import math

logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class Drone:
    def __init__(self, connection_string='COM15', baudrate=57600):
        print('vehicle connecting...')

        # Connecting value
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.vehicle = connect(self.connection_string, wait_ready=True, baud=self.baudrate, timeout=60)

        # Communication
        self.received_data = None
        self.vehicle.add_message_listener('DATA64', self.data64_callback)

        # Position value
        self.init_lat = self.vehicle.location.global_relative_frame.lat
        self.init_lon = self.vehicle.location.global_relative_frame.lon

    def data64_callback(self, vehicle, name, message):
        # Unpacking the received data
        data = [int.from_bytes(message.data[i:i + 4], 'little') for i in range(0, len(message.data), 4)]
        self.received_data = data

    def receiving_data(self):
        return self.received_data

    def close_connection(self):
        self.vehicle.close()

    # 드론 액션 함수

    def arm(self):
        try:
            self.vehicle.channels.overrides['3'] = 1120
            self.vehicle.mode = VehicleMode("STABILIZE")
            time.sleep(0.1)
            self.vehicle.armed = True
            time.sleep(5.6)  # Wait for the drone to be armed

            if self.vehicle.armed:
                print("Vehicle armed")
            else:
                print("Vehicle armed fail")
        except APIException as e:
            print(str(e))

    def takeoff(self, h):
        self.vehicle.mode = VehicleMode("STABILIZE")
        time.sleep(0.1)
        self.vehicle.channels.overrides['3'] = 1300
        time.sleep(3)
        self.vehicle.mode = VehicleMode("GUIDED")

        cmds = self.vehicle.commands
        cmds.download()
        cmds.wait_ready()
        cmds.clear()
        takeoff_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                              mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, h)
        cmds.add(takeoff_cmd)
        cmds.upload()

        self.vehicle.mode = VehicleMode("AUTO")
        time.sleep(0.1)

        while not self.vehicle.armed:
            print("Waiting for arming...")

        print("Taking off!")

        # Monitoring altitude
        while True:
            print(f"Altitude: {self.vehicle.location.global_relative_frame.alt}")
            if self.vehicle.location.global_relative_frame.alt >= h * 0.8:
                print("Reached target altitude")
                break
            time.sleep(0.5)

    def goto_auto(self, x, y, z, velocity=4):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        base_lat = self.init_lat
        base_lon = self.init_lon

        target_lat = base_lat + (y / LATITUDE_CONVERSION)
        target_lon = base_lon + (x / LONGITUDE_CONVERSION)

        self.vehicle.mode = VehicleMode("GUIDED")
        time.sleep(0.1)

        cmds = self.vehicle.commands
        cmds.download()
        cmds.wait_ready()
        cmds.clear()

        goto_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, velocity, 0, target_lat, target_lon, z)
        cmds.add(goto_cmd)
        cmds.upload()

        self.vehicle.mode = VehicleMode("AUTO")

        # 모니터 링
        while math.dist([self.vehicle.location.global_relative_frame.lat,
                         self.vehicle.location.global_relative_frame.lon,
                         self.vehicle.location.global_relative_frame.alt],
                        [target_lat, target_lon, z]) > 0.5:  # Adjust as needed
            print(f"Current Position: {self.vehicle.location.global_relative_frame}")
            time.sleep(0.5)
        print("Reached target position")

    def goto_guided(self, x, y, z, velocity=4):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        base_lat = self.init_lat
        base_lon = self.init_lon

        target_lat = base_lat + (y / LATITUDE_CONVERSION)
        target_lon = base_lon + (x / LONGITUDE_CONVERSION)

        self.vehicle.mode = VehicleMode("GUIDED")
        time.sleep(0.1)

        # Set target location and velocity in GUIDED mode
        location = LocationGlobalRelative(target_lat, target_lon, z)
        self.vehicle.simple_goto(location, groundspeed=velocity)

        # Monitoring the position
        while math.dist([self.vehicle.location.global_relative_frame.lat,
                         self.vehicle.location.global_relative_frame.lon,
                         self.vehicle.location.global_relative_frame.alt],
                        [target_lat, target_lon, z]) > 0.5:  # Adjust as needed
            print(f"Current Position: {self.vehicle.location.global_relative_frame}")
            time.sleep(0.5)
        print("Reached target position")

    def land_by_auto_mode(self):
        self.vehicle.mode = VehicleMode("GUIDED")
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

        while self.vehicle.location.global_relative_frame.alt > 0.1:
            print(f"Altitude: {self.vehicle.location.global_relative_frame.alt}")
            time.sleep(1)

        print("Landed successfully!")

    # # RTL 고도/속도 조절 (추후에 반드시 해야함)

    def return_to_launch(self, h=3, velocity=4):
        target_lat = self.init_lat
        target_lon = self.init_lon

        self.vehicle.mode = VehicleMode("GUIDED")
        time.sleep(0.1)

        cmds = self.vehicle.commands
        cmds.download()
        cmds.wait_ready()
        cmds.clear()

        goto_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, velocity, 0, target_lat, target_lon, h)
        land_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0,
                           self.vehicle.location.global_relative_frame.lat,
                           self.vehicle.location.global_relative_frame.lon, 0)  # 0 for altitude as it's landing
        cmds.add(goto_cmd)
        cmds.add(land_cmd)
        cmds.upload()

        self.vehicle.mode = VehicleMode("AUTO")

        # 모니터 링
        while math.dist([self.vehicle.location.global_relative_frame.lat,
                         self.vehicle.location.global_relative_frame.lon,
                         self.vehicle.location.global_relative_frame.alt],
                        [target_lat, target_lon, h]) > 0.5:  # Adjust as needed
            print(f"Current Position: {self.vehicle.location.global_relative_frame}")
            time.sleep(0.5)
        print("Landed successfully!")

    def close_connection(self):
        self.vehicle.close()


if __name__ == "__main__":
    # 드론 연결
    gt = Drone()

    # 드론 행동 블럭
    try:
        # 입력 (위도, 경도)
        raw_input = input("두 개의 실수를 입력하세요: ")
        nums = [float(num.strip()) for num in raw_input.split(",")]

        # 미션 시작
        if len(nums) == 2:
            print(gt.init_lat, gt.init_lon)
            print(gt.receiving_data())
            time.sleep(1)
            gt.arm()
            gt.takeoff(3)
            gt.land_by_auto_mode()
            # 움직임 사이사이 딜레이 조절 코드 넣기
            # auto vs guided
            # 속도 10m/s 까지 해보기

        else:
            print("정확하게 두 개의 실수를 입력하세요.")

    except ValueError:
        print("올바른 형식의 실수를 입력하세요.")
    except KeyboardInterrupt:
        gt.close_connection()
