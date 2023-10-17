def imm_tracking(data, num_steps=2):
    
    def interpolate_missing_data(data):
        interpolated_data = []
        for i in range(len(data)-1):
            t1, x1, y1, z1 = data[i]
            t2, x2, y2, z2 = data[i+1]
            
            interpolated_data.append((t1, x1, y1, z1))
            
            for t in range(t1+1, t2):
                factor = (t - t1) / (t2 - t1)
                x = x1 + factor * (x2 - x1)
                y = y1 + factor * (y2 - y1)
                z = z1 + factor * (z2 - z1)
                interpolated_data.append((t, x, y, z))
                
        interpolated_data.append(data[-1])
        return interpolated_data
    
    data = interpolate_missing_data(data)
    
    def compute_velocity_and_acceleration(values, times):
        delta_t1 = times[-1] - times[-2]
        delta_t2 = times[-2] - times[-3]
        v_t_1 = (values[-2] - values[-3]) / delta_t2
        v_t = (values[-1] - values[-2]) / delta_t1
        a_t = (v_t - v_t_1) / ((delta_t1 + delta_t2) / 2)
        return v_t, a_t
    
    def predict_next_step(x, vx, ax, delta_t):
        x_cv = x + vx * delta_t
        x_ca = x + vx * delta_t + 0.5 * ax * delta_t**2
        w1, w2 = 0.6, 0.4  # 모델 가중치
        return w1 * x_cv + w2 * x_ca
    
    time_values = [t for t, x, y, z in data]
    x_values = [x for t, x, y, z in data]
    y_values = [y for t, x, y, z in data]
    z_values = [z for t, x, y, z in data]
    
    vx, ax = compute_velocity_and_acceleration(x_values, time_values)
    vy, ay = compute_velocity_and_acceleration(y_values, time_values)
    vz, az = compute_velocity_and_acceleration(z_values, time_values)
    
    predictions = []
    delta_t_pred = time_values[-1] - time_values[-2]  # 예측에 사용할 시간 간격
    
    x_next, y_next, z_next = x_values[-1], y_values[-1], z_values[-1]
    
    for _ in range(num_steps):
        x_next = predict_next_step(x_next, vx, ax, delta_t_pred)
        y_next = predict_next_step(y_next, vy, ay, delta_t_pred)
        z_next = predict_next_step(z_next, vz, az, delta_t_pred)
        
        vx += ax * delta_t_pred
        vy += ay * delta_t_pred
        vz += az * delta_t_pred
        
        predictions.append((x_next, y_next, z_next))
        
    return predictions


# 사용 예시

data = [(1, 0, 0, 0),(2, 1, 1, 1),(3, 2, 2, 2),(4, 3, 3, 3),(5, 4, 4, 4),(6, 5, 5, 5)]

num_steps_to_predict = 2 # 예측할 타임 스텝 수

imm_predictions = imm_tracking(data, num_steps=num_steps_to_predict)
print(imm_predictions)