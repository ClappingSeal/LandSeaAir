import serial
import time
import struct
# 짐벌과 연결된 시리얼 포트와 보레이트를 설정합니다.
ser = serial.Serial('COM11', 115200, timeout=3)
# 연결 확인
if ser.isOpen():
    print("Connection is established!")
else:
    print("Error in serial connection!")
class GimbalController:
    def __init__(self, serial_port, frame_width=850, frame_height=480, max_yaw=135, max_pitch=25, min_pitch=-90):
        self.serial_port = serial_port
        self.current_yaw = 0
        self.current_pitch = 0
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.min_pitch = min_pitch
        self.crc16_tab = [0x0,0x1021,0x2042,0x3063,0x4084,0x50a5,0x60c6,0x70e7,
            0x8108,0x9129,0xa14a,0xb16b,0xc18c,0xd1ad,0xe1ce,0xf1ef,
            0x1231,0x210,0x3273,0x2252,0x52b5,0x4294,0x72f7,0x62d6,
            0x9339,0x8318,0xb37b,0xa35a,0xd3bd,0xc39c,0xf3ff,0xe3de,
            0x2462,0x3443,0x420,0x1401,0x64e6,0x74c7,0x44a4,0x5485,
            0xa56a,0xb54b,0x8528,0x9509,0xe5ee,0xf5cf,0xc5ac,0xd58d,
            0x3653,0x2672,0x1611,0x630,0x76d7,0x66f6,0x5695,0x46b4,
            0xb75b,0xa77a,0x9719,0x8738,0xf7df,0xe7fe,0xd79d,0xc7bc,
            0x48c4,0x58e5,0x6886,0x78a7,0x840,0x1861,0x2802,0x3823,
            0xc9cc,0xd9ed,0xe98e,0xf9af,0x8948,0x9969,0xa90a,0xb92b,
            0x5af5,0x4ad4,0x7ab7,0x6a96,0x1a71,0xa50,0x3a33,0x2a12,
            0xdbfd,0xcbdc,0xfbbf,0xeb9e,0x9b79,0x8b58,0xbb3b,0xab1a,
            0x6ca6,0x7c87,0x4ce4,0x5cc5,0x2c22,0x3c03,0xc60,0x1c41,
            0xedae,0xfd8f,0xcdec,0xddcd,0xad2a,0xbd0b,0x8d68,0x9d49,
            0x7e97,0x6eb6,0x5ed5,0x4ef4,0x3e13,0x2e32,0x1e51,0xe70,
            0xff9f,0xefbe,0xdfdd,0xcffc,0xbf1b,0xaf3a,0x9f59,0x8f78,
            0x9188,0x81a9,0xb1ca,0xa1eb,0xd10c,0xc12d,0xf14e,0xe16f,
            0x1080,0xa1,0x30c2,0x20e3,0x5004,0x4025,0x7046,0x6067,
            0x83b9,0x9398,0xa3fb,0xb3da,0xc33d,0xd31c,0xe37f,0xf35e,
            0x2b1,0x1290,0x22f3,0x32d2,0x4235,0x5214,0x6277,0x7256,
            0xb5ea,0xa5cb,0x95a8,0x8589,0xf56e,0xe54f,0xd52c,0xc50d,
            0x34e2,0x24c3,0x14a0,0x481,0x7466,0x6447,0x5424,0x4405,
            0xa7db,0xb7fa,0x8799,0x97b8,0xe75f,0xf77e,0xc71d,0xd73c,
            0x26d3,0x36f2,0x691,0x16b0,0x6657,0x7676,0x4615,0x5634,
            0xd94c,0xc96d,0xf90e,0xe92f,0x99c8,0x89e9,0xb98a,0xa9ab,
            0x5844,0x4865,0x7806,0x6827,0x18c0,0x8e1,0x3882,0x28a3,
            0xcb7d,0xdb5c,0xeb3f,0xfb1e,0x8bf9,0x9bd8,0xabbb,0xbb9a,
            0x4a75,0x5a54,0x6a37,0x7a16,0xaf1,0x1ad0,0x2ab3,0x3a92,
            0xfd2e,0xed0f,0xdd6c,0xcd4d,0xbdaa,0xad8b,0x9de8,0x8dc9,
            0x7c26,0x6c07,0x5c64,0x4c45,0x3ca2,0x2c83,0x1ce0,0xcc1,
            0xef1f,0xff3e,0xcf5d,0xdf7c,0xaf9b,0xbfba,0x8fd9,0x9ff8,
            0x6e17,0x7e36,0x4e55,0x5e74,0x2e93,0x3eb2,0xed1,0x1ef0
            ]
    def CRC16_cal(self, ptr, len_, crc_init=0):
        crc = crc_init
        for i in range(len_):
            temp = (crc >> 8) & 0xff
            crc = ((crc << 8) ^ self.crc16_tab[ptr[i] ^ temp]) & 0xffff
        return crc

    def set_gimbal_angle(self, yaw, pitch): # 각도 체크섬 생성 및 각도 조종 명령 주기
        cmd_header = b'\x55\x66\x01\x04\x00\x00\x00\x0E'
        yaw_bytes = struct.pack('<h', int(yaw * 10))
        pitch_bytes = struct.pack('<h', int(pitch * 10))
        data_to_checksum = cmd_header + yaw_bytes + pitch_bytes
        calculated_checksum = self.CRC16_cal(data_to_checksum, len(data_to_checksum))
        checksum_bytes = struct.pack('<H', calculated_checksum)
        command = data_to_checksum + checksum_bytes
        self.send_command_to_gimbal(command)

        self.current_yaw = yaw
        self.current_pitch = pitch

    def send_command_to_gimbal(self, command_bytes):
        self.serial_port.write(command_bytes)
        response = self.serial_port.read(10)
        if response:
            print("Received:", response)
        else:
            print("No response from gimbal.")

    def adjust_gimbal_relative_to_current(self, target_x, target_y): # 상대 각도
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2

        diff_x = target_x - center_x
        diff_y = target_y - center_y

        # If the difference is zero, then there's no need to adjust
        if diff_x == 0 and diff_y == 0:
            print("Target is at the center. No adjustment needed.")
            return

        scale_factor_yaw = self.max_yaw / center_x
        scale_factor_pitch = (self.max_pitch - self.min_pitch) / center_y

        yaw_adjustment = self.current_yaw + diff_x * scale_factor_yaw
        pitch_adjustment = self.current_pitch - diff_y * scale_factor_pitch

        yaw_adjustment = max(-self.max_yaw, min(self.max_yaw, yaw_adjustment))
        pitch_adjustment = max(self.min_pitch, min(self.max_pitch, pitch_adjustment))

        self.set_gimbal_angle(yaw_adjustment, pitch_adjustment)

    def adjust_gimbal(self, target_x, target_y): # 절대 각도
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2

        diff_x = target_x - center_x
        diff_y = target_y - center_y

        scale_factor_yaw = self.max_yaw / center_x
        scale_factor_pitch = (self.max_pitch - self.min_pitch) / center_y

        yaw_adjustment = self.current_yaw + diff_x * scale_factor_yaw
        pitch_adjustment = self.current_pitch - diff_y * scale_factor_pitch

        yaw_adjustment = max(-self.max_yaw, min(self.max_yaw, yaw_adjustment))
        pitch_adjustment = max(self.min_pitch, min(self.max_pitch, pitch_adjustment))

        self.set_gimbal_angle(yaw_adjustment, pitch_adjustment)

gimbal = GimbalController(ser)
gimbal.adjust_gimbal_relative_to_current(425, 120) # frame 값 넣기
# gimbal.adjust_gimbal(425, 360) # frame 값 넣기

# gimbal.adjust_gimbal(425, 480) # 아래 보는 frame