from dronekit import Command, VehicleMode, connect
from pymavlink.dialects.v20 import ardupilotmega
import time
import logging

logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class DroneReceiver:
    def __init__(self, connection_string='COM5', baudrate=57600):
        print('vehicle connecting')
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.vehicle = connect(self.connection_string, wait_ready=True, baud=self.baudrate)
        self.vehicle.add_message_listener('DATA64', self.on_data64)
        self.received_data = None

    def on_data64(self, vehicle, name, message):
        if isinstance(message, ardupilotmega.MAVLink_data64_message):
            data = [int.from_bytes(message.data[i:i + 4], 'little') for i in range(0, len(message.data), 4)]
            self.received_data = data

    def get_received_data(self):
        return self.received_data

    def close_connection(self):
        self.vehicle.close()


if __name__ == "__main__":
    drone_receiver = DroneReceiver()
    try:
        while True:
            if drone_receiver.get_received_data():
                a = drone_receiver.get_received_data()
                print(a, a[0])
            time.sleep(1)
    except KeyboardInterrupt:
        drone_receiver.close_connection()
