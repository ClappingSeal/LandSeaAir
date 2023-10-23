import os
import numpy as np
from stable_baselines3 import TD3
from locking_env_TD3_pos_straight_learning import LockingEnv
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # 마름모를 그리기 위한 모듈 추가

# 환경 및 모델 로드
env = LockingEnv()
model = TD3.load("tracking_model_td3_pos_straight_1023")

num_tests = 1
results_dir = 'saved_plots' + date.today().isoformat()

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

for test in range(num_tests):
    obs = env.reset()
    
    done = False
    x_points = [obs[0]]
    y_points = [obs[1]]

    fig = plt.figure(figsize=(32, 24))
    ax = fig.add_subplot(1, 1, 1)

    # 원하는 글씨 크기를 설정
    title_fontsize = 40
    label_fontsize = 50
    legend_fontsize = 50
    rectangle_linewidth = 3  # 원하는 사각형 선 굵기를 설정

    # 원의 중심과 반지름을 정의
    circle_center = (0, 0)
    circle_radius = 5
    circle = patches.Circle(circle_center, circle_radius, fill=True, color=(0, 0, 1, 0.2), edgecolor='blue')
    ax.add_patch(circle)
    f = 0
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info, action, rand = env.step(action)

        obs[0] += rand[0]
        obs[1] += rand[1]

        ax.clear()
        ax.set_xlim(-122, 122)
        ax.set_ylim(-114, 114)

        circle_center = (circle_center[0]-action[0]+rand[0], circle_center[1]-action[1]+rand[1])

        # 직사각형 그리기
        rect_bottomleft_x = circle_center[0] - 42.5
        rect_bottomleft_y = circle_center[1] - 24
        rectangle = patches.Rectangle((rect_bottomleft_x, rect_bottomleft_y), 85, 48, fill=False, color=(1, 0, 0, 0.5), linewidth=rectangle_linewidth, label="camera")
        ax.add_patch(rectangle)

        # 매번 새로 그릴 때마다 원도 다시 추가
        circle = patches.Circle(circle_center, circle_radius, fill=True, color=(0, 0, 1, 0.2), edgecolor='blue')
        ax.add_patch(circle)

        ax.scatter(x_points, y_points, s=50, color=(0, 1, 0, 0.2), marker='o')
        ax.scatter(x_points[-1], y_points[-1], s=100, c='g', marker='o', label="target position")
        x_points.append(x_points[-1] + rand[0]) 
        y_points.append(y_points[-1] + rand[1]) # 불법드론의 바뀐 위치 추가
        ax.set_title(f"Agent Trajectory for Test {test+1} Step {f}", fontsize=title_fontsize)
        ax.set_xlabel("X Coordinate", fontsize=label_fontsize)
        ax.set_ylabel("Y Coordinate", fontsize=label_fontsize)
        ax.legend(fontsize=legend_fontsize)
        fig.savefig(os.path.join(results_dir, f"trajectory_{test+1}_step{f}.png"))
        f += 1
    plt.close(fig)

print("모든 테스트를 시각화하여 저장했습니다!")
