import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


from stable_baselines3.common.results_plotter import load_results, ts2xy


from locking_env_TD3_pos_straight_learning import LockingEnv

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=1000) # window = 5000
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(f"{log_folder}/{date.today().isoformat()}curve.png")


print('All modules imported!')

log_dir = 'log' + date.today().isoformat() +'/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

eval_env = Monitor(LockingEnv(), log_dir)
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=3000,
                             deterministic=True, render=False)

total_timesteps= 100000
print(eval_env.action_space)
n_actions = eval_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", eval_env, action_noise=action_noise, verbose=1)

import time
startTime = time.time()
model.learn(total_timesteps, callback=eval_callback)
endTime = time.time()
print(endTime-startTime)

plot_results(log_dir)
model.save("tracking_model_td3_pos_straight_1023")
