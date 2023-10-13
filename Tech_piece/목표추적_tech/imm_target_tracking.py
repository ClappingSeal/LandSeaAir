import pandas as pd
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of Earth in meters

    return c * r

def transform_coordinates(df, base_lat=35.2265867, base_lon=126.8397070):
    df['x'] = df.apply(lambda row: haversine(base_lon, base_lat, row['Longitude'], base_lat), axis=1)
    df['y'] = df.apply(lambda row: haversine(base_lon, base_lat, base_lon, row['Latitude']), axis=1)
    df.rename(columns={"Altitude": "z", "Time": "time"}, inplace=True)

    return df[['time', 'x', 'y', 'z']]

def imm_tracking(df, num_steps=2):
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
    
    x_values = df['x'].values
    y_values = df['y'].values
    z_values = df['z'].values
    time_values = df['time'].values
    
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