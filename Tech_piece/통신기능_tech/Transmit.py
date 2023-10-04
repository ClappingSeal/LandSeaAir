from pymavlink import mavutil

# UART 연결 생성
master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)

# 데이터 전송
data = [1, 2]
packed_data = bytearray()
for item in data:
    packed_data += item.to_bytes(4, 'little')  # 각 항목을 4바이트로 변환하여 합침

# 데이터를 64바이트로 패딩
while len(packed_data) < 64:
    packed_data += b'\x00'

master.mav.data64_send(0, len(packed_data), packed_data)

# 연결 종료
master.close()
