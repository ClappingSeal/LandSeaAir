from pymavlink import mavutil
import cv2
import threading
import time

class Drone:
    def __init__(self, connection_string='/dev/ttyACM0', baudrate=115200):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.vehicle = mavutil.mavlink_connection(self.connection_string, baud=self.baudrate)
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Error: Couldn't open the camera.")
            return

    def send_data(self, data):
        # Packing Data
        packed_data = bytearray()
        for item in data:
            packed_data += item.to_bytes(4, 'little')

        # 64byte Padding
        while len(packed_data) < 64:
            packed_data += b'\x00'

        # Sending Data
        self.vehicle.mav.data64_send(0, len(packed_data), packed_data)

    def show_camera_stream(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Couldn't read frame.")
                break

            cv2.imshow("Camera Stream", frame) # delete this line to make process quick
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    def close_connection(self):
        self.vehicle.close()


if __name__=='__main__':
    drone = Drone()

    camera_thread = threading.Thread(target=drone.show_camera_stream)
    camera_thread.start()
    while True:
        drone.send_data([123, 425, 234, 212])
        time.sleep(1)
