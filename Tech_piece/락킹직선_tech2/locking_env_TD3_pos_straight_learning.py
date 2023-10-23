import gym
from gym import spaces

import numpy as np
import math
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

class LockingEnv(gym.Env):

    def __init__(self):
        super(LockingEnv, self).__init__()

        self.n_states = 2

        self.agent_pos = np.zeros((self.n_states,), dtype=np.float32)
        
        # 초기 위치
        x_c = 42.5* (2 * np.random.random() - 1)
        y_c = 24* (2 * np.random.random() - 1)
        self.agent_pos[0] = x_c
        self.agent_pos[1] = y_c

        # self.agent_pos /= 100.0

        self.curr_time = 0.0
        self.steps = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2, ))

        low_arr = -0.8*np.ones((self.n_states,),dtype=np.float32)
        high_arr = 0.8*np.ones((self.n_states,),dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low_arr, high=high_arr,
                                        shape=(self.n_states,), dtype=np.float32)
        
        self.straight_x = (2*np.random.random() - 1)*0.5
        self.straight_y = (2*np.random.random() - 1)*0.5
        
        
    def reset(self):
        x_c = 42.5* (2 * np.random.random() - 1)
        y_c = 24* (2 * np.random.random() - 1)
        self.agent_pos[0] = x_c
        self.agent_pos[1] = y_c

        # self.agent_pos /= 100.0

        self.curr_time = 0.0
        self.steps = 0

        return self.agent_pos

    def step(self, action):
        UAV_vel = 5.0
        grid_size = 1.0
        delta = grid_size
        DEL_T = delta/UAV_vel

        speed_magnitude = np.linalg.norm(action)
        if speed_magnitude > 1:
            action = (action / speed_magnitude)

        self.agent_pos[0] += action[0]
        self.agent_pos[1] += action[1]

        # 랜덤 직선 운동
        self.agent_pos[0] -= self.straight_x
        self.agent_pos[1] -= self.straight_y
        rand = [self.straight_x,self.straight_y]
        # 에이전트의 위치가 지정된 경계를 벗어나지 않도록 제한
        self.agent_pos[0] = np.clip(self.agent_pos[0], -42.5, 42.5)
        self.agent_pos[1] = np.clip(self.agent_pos[1], -24, 24)
        done = False
        # reward
        # if abs(self.agent_pos[0] - 16)+ abs(self.agent_pos[1] - 12) < 10:
        #     reward_center = 12
        # else:
        #     reward_center = 0
        self.curr_time += DEL_T

        distance_to_center = np.linalg.norm(self.agent_pos - np.array([0, 0]))

        reward_total = 5 - distance_to_center
        # print(reward_total)
        if reward_total > 0:
            done = True

        if (self.curr_time >= 100.0):
            done = True
        info = {'reward_total': reward_total}

        return self.agent_pos, reward_total, done, info, action, rand


    def render(self, mode='console'):
        pass

    def close(self):
        pass
