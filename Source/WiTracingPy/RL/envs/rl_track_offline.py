import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

import socket
import time

import settings
from networking.udp import UdpSocketReceiver, UdpSocketSender
from networking.udp_socket_client import UdpSocketClient

import json
import random
from numpy.linalg import norm


class RLtrackOfflineEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    # action_list = ["forward", "turnright", "turnleft", "stop"]

    def __init__(self, render_mode=None, Tx_num=9):
        self.observation_space = spaces.Dict(
            {
                "Last_TXs": spaces.Box(-255, 0, shape=(Tx_num,), dtype=int),
                "TXs": spaces.Box(-255, 0, shape=(Tx_num,), dtype=int),
            }
        )
        # self.action_space = spaces.Discrete(4)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.is_receiving = False
        self.tx_readings = dict()

        self.reward = 0
        self.accumulated_reward = 0

        self.last_rx_position = np.array([0.0, 0.0, 90.0])
        self.last_agent_position = np.array([0.0, 2500.0, 90.0])

        self.rx_position = np.array([0.0, 0.0, 90.0])
        self.agent_position = np.array([0.0, 2500.0, 90.0])

        self.df = self.preprocess_data()
        self.step_length = 60
        self.df_idx = 0
        self.move_scale = 10

        self.movement = [0.0, 0.0]

        self.last_txs = None

    def ab(self, df):
        return df.values

    def preprocess_data(self):
        df = pd.read_csv('raw.csv')
        df = df.groupby(['timestamp', 'x', 'y', 'z'])['rssi'].apply(self.ab)
        df = df.reset_index()

        return df


    def get_reward(self, vector_x, vector_y):
        agent_move = np.array([vector_x * self.move_scale, vector_y * self.move_scale])
        # diff = self.movement - agent_move
        cosine = np.dot(self.movement, agent_move) / (norm(self.movement) * norm(agent_move))

        move_length = np.linalg.norm(self.movement)
        action_length = np.linalg.norm(agent_move)
        distance = -abs(move_length - action_length)

        # reward = cosine + distance
        reward = cosine

        return reward


    def _get_info(self):
        info = {
            "Movement": self.movement
        }
        return info

    def _get_obs(self):
        last_tx = self.df.iloc[self.df_idx].values[4:][0].astype(np.int32)
        last_location = self.df.iloc[self.df_idx].values[1:3]
        tx = self.df.iloc[self.df_idx + self.step_length].values[4:][0].astype(np.int32)
        location = self.df.iloc[self.df_idx + self.step_length].values[1:3]
        self.movement = location - last_location
        self.df_idx += self.step_length
        obs = {
            "Last_TXs": last_tx,
            "TXs": tx,
        }
        return obs

    def reset(self, seed=None, options=None):
        print("Reseting Env ...")
        super().reset(seed=seed)
        random_start = random.randint(0, 60)
        self.df_idx = random_start

        last_tx = self.df.iloc[self.df_idx].values[4:][0].astype(np.int32)
        last_location = self.df.iloc[self.df_idx].values[1:3]
        tx = self.df.iloc[self.df_idx + self.step_length].values[4:][0].astype(np.int32)
        location = self.df.iloc[self.df_idx + self.step_length].values[1:3]
        self.movement = location - last_location
        obs = {
            "Last_TXs": last_tx,
            "TXs": tx,
        }
        self.reward = 0
        self.accumulated_reward = 0
        info = self._get_info()

        self.df_idx += self.step_length

        return obs, info

    def step(self, action):
        action = action[0]
        vector_x = action.tolist()[0]
        vector_y = action.tolist()[1]
        # print("action sent!")
        observation = self._get_obs()
        reward = self.get_reward(vector_x, vector_y)
        self.accumulated_reward += reward
        # print(reward)
        terminated = False
        truncated = False

        # if self.accumulated_reward < -2000:
        #     truncated = True
        # else:
        #     truncated = False

        if self.df_idx + self.step_length >= self.df.shape[0]:
            truncated = True

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
