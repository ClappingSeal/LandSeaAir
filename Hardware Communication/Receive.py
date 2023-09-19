from pymavlink import mavutil

master = mavutil.mavlink_connection('COM4', baud=57600)

while True:
    msg = master.recv_match(blocking=True)

    if msg.get_type() == 'DATA64':
        # 패킹된 데이터 처리 과정
        received_data = msg.data
        numbers = [int.from_bytes(received_data[i:i + 4], 'little') for i in range(0, len(received_data), 4)]
        numbers = [num for num in numbers if num != 0]
        print("Received data:", numbers)

# 이 코드에서 종료는 실행되지 않음
master.close()
# 진혁시치
