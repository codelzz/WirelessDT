import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import torch

import socket
import time

import settings
from networking.udp import UdpSocketReceiver, UdpSocketSender
from networking.udp_socket_client import UdpSocketClient

import json
import random
from numpy.linalg import norm


class RLfuseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode=None):
        self.imu_df = pd.DataFrame()
        self.cam_df = pd.DataFrame()
        self.wifi_df = pd.DataFrame()
        self.cam_unique_timestamps = []
        self.imu_unique_timestamps = []

        # reward related
        self.label_list = []

        self.max_pedestrian_detections = 10
        self.timestamp_length = 50
        self.imu_target = 'BP_TargetAI_C_0'
        imu_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(self.timestamp_length, 6), dtype=np.float32)
        visual_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.timestamp_length, self.max_pedestrian_detections * 2), dtype=np.float32)

        self.observation_space = gym.spaces.Tuple((imu_space, visual_space))
        self.action_space = gym.spaces.Discrete(self.max_pedestrian_detections)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.curr_timestamp_idx = 0

    def load_data(self, wifi_df, cam_df, imu_df):
        self.wifi_df = wifi_df
        self.cam_df = cam_df
        self.imu_df = imu_df[imu_df['IMU_ID'] == self.imu_target]

        self.cam_unique_timestamps = self.cam_df['timestamp'].drop_duplicates().tolist()
        self.imu_unique_timestamps = self.imu_df['timestamp'].drop_duplicates().tolist()

        # First, create a copy of the sliced DataFrame
        self.imu_df = self.imu_df.copy()

        self.imu_df.loc[:, 'acceleration_x'] = (self.imu_df.loc[:, 'acceleration_x'] - self.imu_df.loc[:, 'acceleration_x'].mean()) / \
                                        self.imu_df.loc[:, 'acceleration_x'].std()
        self.imu_df.loc[:, 'acceleration_y'] = (self.imu_df.loc[:, 'acceleration_y'] - self.imu_df.loc[:, 'acceleration_y'].mean()) / \
                                        self.imu_df.loc[:, 'acceleration_y'].std()
        self.imu_df.loc[:, 'orientation_z'] = (self.imu_df.loc[:, 'orientation_z'] - self.imu_df.loc[:, 'orientation_z'].mean()) / \
                                        self.imu_df.loc[:, 'orientation_z'].std()

    def find_target_label(self, matrix, target):
        positions = []
        for i, row in enumerate(matrix):
            found_in_row = False
            for j, element in enumerate(row):
                if element == target:
                    positions.append(j)
                    found_in_row = True
            if not found_in_row:
                positions.append(-1)
        return positions

    def get_reward(self, action_label):
        reward = []
        target_label = self.find_target_label(self.label_list, self.imu_target)
        for action, target in zip(action_label, target_label):
            if action == target:
                reward.append(1)
            else:
                reward.append(0)
        return reward

    def _get_info(self):
        pass

    def shuffle_data(self, df):
        return df.sample(frac=1).reset_index(drop=True)

    def _get_obs(self, curr_timestamp_idx):
        label_list = []
        x_list = []
        y_list = []
        imu_list = []

        for i in range(curr_timestamp_idx, curr_timestamp_idx + self.timestamp_length):
            cam_ts = self.cam_unique_timestamps[i]
            vis_data = self.cam_df[self.cam_df['timestamp'] == cam_ts]
            vis_data = vis_data[vis_data['los'] == True]

            vis_data = self.shuffle_data(vis_data)

            label_list.append(vis_data['Ped_ID'].tolist())

            xl = vis_data['x'].tolist()
            xl += [0.0] * (self.max_pedestrian_detections - len(xl))
            x_list.append(xl)
            yl = vis_data['y'].tolist()
            yl += [0.0] * (self.max_pedestrian_detections - len(yl))
            y_list.append(yl)

            imu_ts = self.imu_unique_timestamps[i]
            imu_data = self.imu_df[self.imu_df['timestamp'] == imu_ts]
            imu_list.append([imu_data['acceleration_x'].tolist()[0],
                             imu_data['acceleration_y'].tolist()[0],
                             imu_data['acceleration_z'].tolist()[0],
                             imu_data['orientation_x'].tolist()[0],
                             imu_data['orientation_y'].tolist()[0],
                             imu_data['orientation_z'].tolist()[0]])
        vis_list = np.concatenate([x_list, y_list], axis=1).tolist()

        imu_list = np.asarray(imu_list, dtype=np.float32)
        vis_list = np.asarray(vis_list, dtype=np.float32)
        self.label_list = label_list

        return vis_list, imu_list, label_list

    def reset(self, seed=None, options=None):
        self.curr_timestamp_idx = random.randint(0, 100)
        vis_list, imu_list, label_list = self._get_obs(self.curr_timestamp_idx)

        obs = (imu_list, vis_list)
        info = {
            "label": label_list
        }
        return obs, info

    def step(self, action):
        highest_probabilities = np.max(action, axis=-1).squeeze()
        max_indices = np.squeeze(np.argmax(action, axis=-1))

        # Replace indices with -1 where the highest probability is lower than 0.5
        modified_indices = np.where(highest_probabilities >= 0.7, max_indices, -1)

        # Convert the modified_indices numpy array to a list
        modified_indices_list = modified_indices.tolist()

        reward = self.get_reward(modified_indices_list)

        terminated = False
        truncated = False

        self.curr_timestamp_idx += 1
        vis_list, imu_list, label_list = self._get_obs(self.curr_timestamp_idx)

        observation = (imu_list, vis_list)
        info = {
            "label": label_list
        }

        if self.curr_timestamp_idx + self.timestamp_length >= len(self.imu_unique_timestamps):
            truncated = True

        return observation, reward, terminated, truncated, info

    def get_evaluate_obs(self, timestep_idx):
        vis_list, imu_list, label_list = self._get_obs(timestep_idx)
        obs = (imu_list, vis_list)
        info = {
            "label": label_list
        }
        return obs, info

    def evaluate_action(self, action):
        highest_probabilities = np.max(action, axis=-1).squeeze()
        max_indices = np.squeeze(np.argmax(action, axis=-1))

        # Replace indices with -1 where the highest probability is lower than 0.5
        modified_indices = np.where(highest_probabilities >= 0.7, max_indices, -1)

        # Convert the modified_indices numpy array to a list
        modified_indices_list = modified_indices.tolist()
        reward = self.get_reward(modified_indices_list)
        return modified_indices_list, reward
    def render(self):
        pass

    def close(self):
        pass
