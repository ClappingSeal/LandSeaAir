import os
import numpy as np
from stable_baselines3 import TD3
from locking_env_TD3_pos_straight_learning import LockingEnv

env = LockingEnv()
model = TD3.load("tracking_model_td3_pos_straight_0920_2")
model = TD3.load("tracking_model_td3_pos_straight_1023")

# obs = env.reset()
def locking_func(x,y):
    obs = np.array([x, y])

    action, _ = model.predict(obs)

    return -action


action = locking_func(-80, 80)
print(action)