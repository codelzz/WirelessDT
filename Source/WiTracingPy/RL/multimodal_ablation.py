import gymnasium as gym
import os
import torch
import pandas as pd
os.chdir('..')
import RL
from RL.algorithm.rlfuse_ppo import Rlfuse_ppo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import seaborn as sns

import logging
import sys
import datetime

CUR_IDX = 5450
def evaluate_model(env, CUR_IDX):
    obs, info = env.get_evaluate_obs(CUR_IDX)
    action = agent_eval.evaluate_action(obs)
    action_idx, reward = env.evaluate_action(action)
    return obs,  info, action_idx, reward

# obs, info, action_idx, reward = evaluate_model(CUR_IDX)
# wifi_names = obs[0]
# wifi_rssis = obs[1]
# imu_vis = obs[2]
# cam_vis = obs[3]

wifi_df = pd.read_csv('../Services/data/jew_train/raw.csv')
cam_df = pd.read_csv('../Services/data/jew_train/cam_raw.csv')
imu_df = pd.read_csv('../Services/data/jew_train/imu_raw.csv')

timesteps = [10,20,30, 40, 50, 80, 100, 150, 200, 300, 450]

# To use differen't log level for file and console
timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
filename=f'tmp5a_{timestamp}.log'
formatter = logging.Formatter('[%(asctime)s] %(name)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')

file_handler = logging.FileHandler(filename=filename)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

# The handlers have to be at a root level since they are the final output
logging.basicConfig(
    level=logging.DEBUG,
    format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[
        file_handler,
        stream_handler
    ]
)

logger = logging.getLogger(__name__)

success_rate_list = []

for timestep in timesteps:
    env = gym.make('RL/RLfuse-v0')
    env.load_data(wifi_df, cam_df, imu_df, timelength=timestep)
    agent_eval = Rlfuse_ppo(env, logger, load_weight=True)

    CUR_IDX_list = np.random.randint(0, 34000, size=10)
    success_list = []
    for i, idx in enumerate(CUR_IDX_list):
        obs, info, action_idx, reward = evaluate_model(env, idx)
        print(i, idx, sum(reward))
        if sum(reward) / len(reward) >= 0.6:
            success_list.append(1)
        else:
            success_list.append(0)
    success_rate = sum(success_list) / 10
    print(success_rate)

    success_rate_list.append(success_rate)

print(success_rate_list)
