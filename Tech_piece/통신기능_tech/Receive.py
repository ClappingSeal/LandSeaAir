from dronekit import Command, VehicleMode, connect
from pymavlink.dialects.v20 import ardupilotmega
import time
import math
import threading
import logging

logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class Drone:
    def __init__(self, connection_string='COM5', baudrate=57600):
        print("vehicle connecting...")
        self.vehicle = connect(connection_string, wait_ready=True, baud=baudrate)
        self.LATITUDE_CONVERSION = 111000
        self.LONGITUDE_CONVERSION = 88.649 * 1000
        self.vehicle.add_message_listener('DATA64', self.on_data64)

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
        self.vehicle.channels.overrides['3'] = 1000
        self.set_flight_mode_by_pwm(1000)  # pwm signal for Stabilize mode
        time.sleep(1)
        self.vehicle.armed = True
        time.sleep(3)  # Wait for the drone to be armed

    def takeoff(self, h):
        self.set_flight_mode_by_pwm(1000)
        time.sleep(0.1)
        self.vehicle.channels.overrides['3'] = 1600
        time.sleep(2)

        cmds = self.vehicle.commands
        cmds.download()
        cmds.wait_ready()
        cmds.clear()
        takeoff_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                              mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, h)
        cmds.add(takeoff_cmd)
        cmds.upload()

        self.vehicle.mode = VehicleMode("AUTO")
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
            time.sleep(1)

    def goto(self, x, y, z, speed):
        target_lat = self.vehicle.location.global_frame.lat + (y / self.LATITUDE_CONVERSION)
        target_lon = self.vehicle.location.global_frame.lon + (x / self.LONGITUDE_CONVERSION)

        cmds = self.vehicle.commands
        cmds.download()
        cmds.wait_ready()
        cmds.clear()

        # Set the speed
        change_speed_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                                   mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0, 1, speed, -1, 0, 0, 0, 0, 0)
        cmds.add(change_speed_cmd)

        # Set the waypoint
        goto_cmd = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0,
                           target_lat, target_lon, z)

        cmds.add(goto_cmd)
        cmds.upload()

        self.vehicle.mode = VehicleMode("AUTO")

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

    def on_data64(self, message):
        if isinstance(message, ardupilotmega.MAVLink_data64_message):
            data = [int.from_bytes(message.data[i:i + 4], 'little') for i in range(0, len(message.data), 4)]
            print("Received data:", data)

    def close(self):
        self.vehicle.close()


if __name__ == "__main__":
    drone = Drone()

    try:
        drone.arm()
        time.sleep(100)

    except KeyboardInterrupt:
        drone.close()