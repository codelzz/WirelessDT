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
        self.wifi_unique_timestamps = []
        self.wifi_unique_names = []

        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

        # reward related
        self.label_list = []

        self.max_pedestrian_detections = 10
        self.timestamp_length = 150
        self.imu_target = 'BP_TargetAI_C_0'
        imu_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.timestamp_length, 6), dtype=np.float32)
        visual_space = gym.spaces.Box(low=0.0, high=1.0,
                                      shape=(self.timestamp_length, self.max_pedestrian_detections, 2),
                                      dtype=np.float32)
        wifi_name_space = gym.spaces.Box(low=0, high=1000, shape=(self.timestamp_length, 20), dtype=np.int32)
        wifi_rssi_space = gym.spaces.Box(low=0, high=255, shape=(self.timestamp_length, 20), dtype=np.int32)

        self.observation_space = gym.spaces.Tuple((wifi_name_space, wifi_rssi_space, imu_space, visual_space))
        self.action_space = gym.spaces.Discrete(self.max_pedestrian_detections)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.curr_timestamp_idx = 0

        self.conf_score = 0.15

    def load_data(self, wifi_df, cam_df, imu_df, timelength=150):
        self.timestamp_length = timelength
        # if timelength % 50 == 0:
        #     self.timestamp_length = timelength
        # else:
        #     print("timelength not valid!")
        self.wifi_df = wifi_df
        self.cam_df = cam_df
        self.imu_df = imu_df[imu_df['IMU_ID'] == self.imu_target].reset_index()

        self.cam_unique_timestamps = self.cam_df['timestamp'].drop_duplicates().tolist()
        self.imu_unique_timestamps = self.imu_df['timestamp'].drop_duplicates().tolist()
        self.wifi_unique_timestamps = self.wifi_df['timestamp'].drop_duplicates().tolist()
        self.wifi_unique_names = self.wifi_df['tx'].drop_duplicates().tolist()
        self.wifi_unique_names.append('N/A')

        # First, create a copy of the sliced DataFrame
        self.imu_df = self.imu_df.copy()
        self.wifi_df = self.wifi_df.copy()

        def tolist(df):
            return df.tolist()

        def pad_list(rssi_list, target_length=20, padding_value=-255):
            padded_list = rssi_list.copy()
            while len(padded_list) < target_length:
                padded_list.append(padding_value)
            padded_list = padded_list[:20]
            return padded_list

        self.wifi_df = self.wifi_df.groupby(['timestamp', 'x', 'y', 'z'])[['rssi', 'tx']].agg(tolist).reset_index()
        # Get the TXName string from the data
        for idx, _ in self.wifi_df.iterrows():
            tx_names = self.wifi_df.iloc[idx]['tx']
            tx_rssis = self.wifi_df.iloc[idx]['rssi']

            # Pad each string with spaces to make them have equal length
            # padded_tx_names = [s.ljust(self.pad_length, self.pad_value) for s in tx_names]
            padded_tx_rssis = pad_list(tx_rssis)

            while len(tx_names) < 20:
                tx_names.append('N/A')
            if len(tx_names) >= 20:
                tx_names = tx_names[:20]
            tx_names = ' '.join(tx_names)

            self.wifi_df.at[idx, 'rssi'] = padded_tx_rssis

            # tx_name_ids = torch.tensor(
            #     self.tokenizer.encode(tx_names, padding='max_length', max_length=1024, add_special_tokens=True))

            # Create a dictionary to map each name to its index in self.wifi_unique_names
            name_to_index = {name: index for index, name in enumerate(self.wifi_unique_names)}
            # Iterate through the DataFrame and convert the concatenated names into a list of indexes
            names = self.wifi_df.iloc[idx]['tx']
            index_list = [name_to_index[name] for name in names if name in name_to_index]
            self.wifi_df.at[idx, 'tx'] = index_list
            # self.wifi_df.at[idx, 'tx'] = tx_name_ids.tolist()

        self.imu_df['acceleration_x'] = self.imu_df['acceleration_x'].clip(-2000, 2000)
        self.imu_df['acceleration_y'] = self.imu_df['acceleration_y'].clip(-2000, 2000)
        self.imu_df['orientation_z'] = self.imu_df['orientation_z'].clip(-500, 500)
        self.imu_df.loc[:, 'acceleration_x'] = (self.imu_df.loc[:, 'acceleration_x'] - self.imu_df.loc[:,
                                                                                       'acceleration_x'].min()) / (
                                                           self.imu_df.loc[:, 'acceleration_x'].max() - self.imu_df.loc[
                                                                                                        :,
                                                                                                        'acceleration_x'].min())
        self.imu_df.loc[:, 'acceleration_y'] = (self.imu_df.loc[:, 'acceleration_y'] - self.imu_df.loc[:,
                                                                                       'acceleration_y'].min()) / (
                                                           self.imu_df.loc[:, 'acceleration_y'].max() - self.imu_df.loc[
                                                                                                        :,
                                                                                                        'acceleration_y'].min())
        self.imu_df.loc[:, 'orientation_z'] = (self.imu_df.loc[:, 'orientation_z'] - self.imu_df.loc[:,
                                                                                       'orientation_z'].min()) / (
                                                           self.imu_df.loc[:, 'orientation_z'].max() - self.imu_df.loc[
                                                                                                        :,
                                                                                                        'orientation_z'].min())
        # self.imu_df.loc[:, 'acceleration_x'] = (self.imu_df.loc[:, 'acceleration_x'] - self.imu_df.loc[:,
        #                                                                                'acceleration_x'].mean()) / \
        #                                        self.imu_df.loc[:, 'acceleration_x'].std()
        # self.imu_df.loc[:, 'acceleration_y'] = (self.imu_df.loc[:, 'acceleration_y'] - self.imu_df.loc[:,
        #                                                                                'acceleration_y'].mean()) / \
        #                                        self.imu_df.loc[:, 'acceleration_y'].std()
        # self.imu_df.loc[:, 'orientation_z'] = (self.imu_df.loc[:, 'orientation_z'] - self.imu_df.loc[:,
        #                                                                              'orientation_z'].mean()) / \
        #                                       self.imu_df.loc[:, 'orientation_z'].std()

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

    def reward_add_bias(self, reward):
        # Calculate the matching probability
        prob = np.mean(reward)

        # Calculate the entropy of the matching probabilities
        entropy = np.clip(np.log(prob + 1e-9), a_min=-5, a_max=None)

        # Add the entropy bonus to the reward
        reward = np.array(reward) + entropy

        return prob, reward.tolist()

    def _get_info(self):
        pass

    def shuffle_data(self, df):
        return df.sample(frac=1).reset_index(drop=True)

    # def _get_obs(self, curr_timestamp_idx):
    #     label_list = []
    #     x_list = []
    #     y_list = []
    #     imu_list = []
    #
    #     for i in range(curr_timestamp_idx, curr_timestamp_idx + self.timestamp_length):
    #         cam_ts = self.cam_unique_timestamps[i]
    #         vis_data = self.cam_df[self.cam_df['timestamp'] == cam_ts]
    #         vis_data = vis_data[vis_data['los'] == True]
    #
    #         # vis_data = self.shuffle_data(vis_data)
    #
    #         label_list.append(vis_data['Ped_ID'].tolist())
    #
    #         xl = vis_data['x'].tolist()
    #         xl += [0.0] * (self.max_pedestrian_detections - len(xl))
    #         x_list.append(xl)
    #         yl = vis_data['y'].tolist()
    #         yl += [0.0] * (self.max_pedestrian_detections - len(yl))
    #         y_list.append(yl)
    #
    #         imu_ts = self.imu_unique_timestamps[i]
    #         imu_data = self.imu_df[self.imu_df['timestamp'] == imu_ts]
    #         imu_list.append([imu_data['acceleration_x'].tolist()[0],
    #                          imu_data['acceleration_y'].tolist()[0],
    #                          imu_data['acceleration_z'].tolist()[0],
    #                          imu_data['orientation_x'].tolist()[0],
    #                          imu_data['orientation_y'].tolist()[0],
    #                          imu_data['orientation_z'].tolist()[0]])
    #     vis_list = np.concatenate([x_list, y_list], axis=1).tolist()
    #
    #     imu_list = np.asarray(imu_list, dtype=np.float32)
    #     vis_list = np.asarray(vis_list, dtype=np.float32)
    #     self.label_list = label_list
    #
    #     return vis_list, imu_list, label_list

    def _get_obs(self, curr_timestamp_idx):
        label_list = []
        x_list = []
        y_list = []
        imu_list = []
        vis_lists = []
        wifi_name_lists = []
        wifi_rssi_lists = []

        cam_ts = self.cam_unique_timestamps[curr_timestamp_idx]
        vis_data = self.cam_df[self.cam_df['timestamp'] == cam_ts]
        vis_data = vis_data[vis_data['los'] == True]
        target_row = 0
        if vis_data.shape[0] >= 6:
            target_row = random.randint(0, 6)
        else:
            target_row = 0

        # inteval = int(self.timestamp_length / 50)
        inteval = 1

        def find_ts(cam_ts):
            self.imu_df['timestamp'] = pd.to_datetime(self.imu_df['timestamp'], unit='ms')
            self.wifi_df['timestamp'] = pd.to_datetime(self.wifi_df['timestamp'], unit='ms')
            given_timestamp = pd.Timestamp(cam_ts, unit='ms')

            # Compute absolute difference between each timestamp and the given timestamp
            # Find index of minimum time difference
            time_diff = abs(self.imu_df['timestamp'] - given_timestamp)
            imu_index = time_diff.idxmin()
            time_diff_wifi = abs(self.wifi_df['timestamp'] - given_timestamp)
            wifi_index = time_diff_wifi.idxmin()

            return imu_index, wifi_index

        imu_idx, wifi_idx = find_ts(cam_ts)

        # imu_ts = self.imu_df.iloc[[imu_idx]]
        # wifi_ts = self.wifi_df.iloc[[wifi_idx]]

        for i in range(curr_timestamp_idx, curr_timestamp_idx + self.timestamp_length, inteval):
            cam_ts = self.cam_unique_timestamps[i]
            wifi_ts = self.wifi_unique_timestamps[i]

            vis_data = self.cam_df[self.cam_df['timestamp'] == cam_ts]
            vis_data = vis_data[vis_data['los'] == True]


            def swap_rows(df, row1, row2):
                df.iloc[row1], df.iloc[row2] = df.iloc[row2].copy(), df.iloc[row1].copy()
                return df
            # TODO Not perfect way to conditionly swap rows
            if target_row != 0 and target_row < len(vis_data):
                vis_data = swap_rows(vis_data, 0, target_row)

            # vis_data = self.shuffle_data(vis_data)

            label_list.append(vis_data['Ped_ID'].tolist())

            xl = vis_data['x'].tolist()
            xl += [0.0] * (self.max_pedestrian_detections - len(xl))
            xl = np.array(xl).reshape(-1, 1)
            # x_list.append(xl)
            yl = vis_data['y'].tolist()
            yl += [0.0] * (self.max_pedestrian_detections - len(yl))
            yl = np.array(yl).reshape(-1, 1)
            # y_list.append(yl)

            vis_list = np.hstack((xl, yl))
            vis_lists.append(vis_list)
            # imu_ts = self.imu_unique_timestamps[i]
            # imu_data = self.imu_df[self.imu_df['timestamp'] == imu_ts]
            imu_data = self.imu_df.iloc[[imu_idx]]
            imu_list.append([imu_data['acceleration_x'].tolist()[0],
                             imu_data['acceleration_y'].tolist()[0],
                             imu_data['acceleration_z'].tolist()[0],
                             imu_data['orientation_x'].tolist()[0],
                             imu_data['orientation_y'].tolist()[0],
                             imu_data['orientation_z'].tolist()[0]])

            # wifi_data = self.wifi_df[self.wifi_df['timestamp'] == wifi_ts]
            wifi_data = self.wifi_df.iloc[[wifi_idx]]
            wifi_name = wifi_data['tx'].tolist()[0]
            wifi_rssi = wifi_data['rssi'].tolist()[0]

            wifi_name_lists.append(wifi_name)
            wifi_rssi_lists.append(wifi_rssi)

            imu_idx += 1
            wifi_idx += 1
        # vis_list = np.concatenate([x_list, y_list], axis=1).tolist()

        imu_list = np.asarray(imu_list, dtype=np.float32)
        vis_lists = np.asarray(vis_lists, dtype=np.float32)
        wifi_name_lists = np.asarray(wifi_name_lists, dtype=np.int32)
        wifi_rssi_lists = np.abs(np.asarray(wifi_rssi_lists, dtype=np.int32))

        self.label_list = label_list

        return wifi_name_lists , wifi_rssi_lists, vis_lists, imu_list, label_list

    def reset(self, seed=None, options=None):
        self.curr_timestamp_idx = random.randint(0, 25000)
        wifi_name_lists , wifi_rssi_lists, vis_list, imu_list, label_list = self._get_obs(self.curr_timestamp_idx)

        obs = (wifi_name_lists , wifi_rssi_lists, imu_list, vis_list)
        info = {
            "label": label_list
        }
        return obs, info

    def step(self, action):
        highest_probabilities = np.max(action, axis=-1).squeeze()
        max_indices = np.squeeze(np.argmax(action, axis=-1))

        # Replace indices with -1 where the highest probability is lower than 0.5
        modified_indices = np.where(highest_probabilities >= self.conf_score, max_indices, -1)

        # Convert the modified_indices numpy array to a list
        modified_indices_list = modified_indices.tolist()

        reward = self.get_reward(modified_indices_list)
        # prob, reward = self.reward_add_bias(reward)
        prob = np.mean(reward)
        terminated = False
        truncated = False

        # self.curr_timestamp_idx += 1
        self.curr_timestamp_idx = random.randint(0, 25000)
        wifi_name_lists , wifi_rssi_lists, vis_list, imu_list, label_list = self._get_obs(self.curr_timestamp_idx)

        observation = (wifi_name_lists , wifi_rssi_lists, imu_list, vis_list)
        info = {
            "label": label_list
        }

        if self.curr_timestamp_idx + self.timestamp_length >= len(self.imu_unique_timestamps):
            truncated = True

        return observation, (prob, reward), terminated, truncated, info

    def get_evaluate_obs(self, timestep_idx):
        wifi_name_lists , wifi_rssi_lists, vis_list, imu_list, label_list = self._get_obs(timestep_idx)
        obs = (wifi_name_lists , wifi_rssi_lists, imu_list, vis_list)
        info = {
            "label": label_list
        }
        return obs, info

    def evaluate_action(self, action):
        highest_probabilities = np.max(action, axis=-1).squeeze()
        max_indices = np.squeeze(np.argmax(action, axis=-1))

        # Replace indices with -1 where the highest probability is lower than 0.5
        modified_indices = np.where(highest_probabilities >= self.conf_score, max_indices, -1)

        # Convert the modified_indices numpy array to a list
        modified_indices_list = modified_indices.tolist()
        reward = self.get_reward(modified_indices_list)
        # prob, bias_reward = self.reward_add_bias(reward)
        return modified_indices_list, reward

    def render(self):
        pass

    def close(self):
        pass
