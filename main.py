from pymavlink import mavutil
import cv2

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

    def arm(self):
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,  
            1,  # 1 to arm
            0, 0, 0, 0, 0, 0  
        )

    def show_camera_stream(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Couldn't read frame.")
                break

            cv2.imshow("Camera Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    def close_connection(self):
        self.vehicle.close()


if __name__=='__main__':
    drone = Drone()
    drone.send_data([123,425,234,212])
    drone.show_camera_stream()

