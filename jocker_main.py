from dronekit import connect
import cv2
import logging
import numpy as np

logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class Drone:
    def __init__(self, connection_string='/dev/ttyACM0', baudrate=115200):
        # Connecting value
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.vehicle = connect(self.connection_string, wait_ready=False, baud=self.baudrate, timeout=100)

        # Camera
        self.camera = cv2.VideoCapture(0)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', fourcc, 20.0, (480, 640))
        if not self.camera.isOpened():
            print("Error: Couldn't open the camera.")
            return

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

    # end
    def close_connection(self):
        self.vehicle.close()

    def detect_drones(self, image):
        def is_similar_points(points, threshold):
            if len(points) < 2:
                return False
            distances = np.linalg.norm(np.array(points) - np.array(points[0]), axis=1)
            return np.all(distances < threshold)

        CONFIDENCE_THRESHOLD = 0.6
        labels = ['hybrid', 'fixed', 'quadcopter']
        recent_centers = []
        max_centers = 3
        similarity_threshold = 50

        # 이미지 사이즈 조정
        resized_frame = cv2.resize(image, (640, 480))  # 예시로 640x480으로 조정
        frame = cv2.rotate(resized_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = frame.shape[:2]

        detection = model(frame, verbose=False)[0]
        for data in detection.boxes.data.tolist():
            confidence = float(data[4])
            if confidence > CONFIDENCE_THRESHOLD:
                xmin, ymin, xlen, ylen = int(data[0]), int(data[1]), int(data[2]) - int(data[0]), int(data[3]) - int(
                    data[1])
                if xmin < 0 or ymin < 0 or xmin + xlen > w or ymin + ylen > h or xlen < 10 or ylen < 10:
                    continue

                rotated_center_x = h - (ymin + ylen // 2)
                rotated_center_y = xmin + xlen // 2
                X = rotated_center_y
                Y = rotated_center_x

                cv2.rectangle(frame, (xmin, ymin), (xmin + xlen, ymin + ylen), (0, 255, 0), 2)
                label_index = int(data[5])
                drone_type = labels[label_index] if label_index < len(labels) else None
                if drone_type:
                    cv2.putText(frame, drone_type, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                center_x, center_y = xmin + xlen // 2, ymin + ylen // 2
                recent_centers.append((center_x, center_y))
                if len(recent_centers) > max_centers:
                    recent_centers.pop(0)

                if is_similar_points(recent_centers, similarity_threshold):
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # 바운딩 박스의 가로, 세로와 함께 기종도 반환
                return X, Y, xlen, ylen, drone_type, frame

        # 드론이 감지되지 않은 경우
        return 240, 320, 0, 0, None, frame

    def detect_dark_objects(self, image, area_threshold=0):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        largest_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_contour = contour
                print(largest_contour)

        if largest_contour is not None and largest_area > area_threshold:
            M = cv2.moments(largest_contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # 센터에 빨간점 찍기
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
            return cx, cy, largest_area, image
        else:
            return 240, 320, 0, image


if __name__ == '__main__':
    drone = Drone()

    try:
        while True:
            ret, frame = drone.camera.read()
            out = cv2.VideoWriter('output.avi', drone.fourcc, 20.0, (640, 480))

            if not ret:
                print("not detected")

            X, Y, area, updated_frame = drone.detect_dark_objects(frame)
            print(f"X: {X}, Y: {Y}, Area: {area}")

            # cv2.imshow('Drone Detection', updated_frame)
            out.write(updated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        drone.camera.release()
        drone.out.release()
        cv2.destroyAllWindows()
