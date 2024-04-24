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
            pㄹ
