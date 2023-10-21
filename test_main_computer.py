# 메인 컴퓨터에서
def send_data_to_rpi(self, data):
    # Packing Data
    packed_data = bytearray()
    for item in data:
        packed_data += item.to_bytes(4, 'little')

    # 64byte Padding
    while len(packed_data) < 64:
        packed_data += b'\x00'

    # Sending Data
    self.vehicle.mav.data64_send(0, len(packed_data), packed_data)

# 데이터 보내기
drone.send_data_to_rpi([567, 890, 345, 678])
