import gymnasium as gym
import numpy as np
import torch

import RL

import socket
import time

import settings

import json
import random

from agent.rltrack_agent import Rltrack_Agent
from networking.udp_socket_client import UdpSocketClient

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.spatial.distance import cosine

if __name__ == "__main__":
    # env = gym.make('RL/RLTrack-v0')
    # wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
    # total_num_episodes = int(5e2)
    # obs_space_dims = env.observation_space["TXs"].shape[0]
    # action_space_dims = env.action_space.shape[0]
    # print(obs_space_dims)
    # print(action_space_dims)
    #
    # rewards_over_seeds = []
    #
    # for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    #     # set seed
    #     torch.manual_seed(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)

    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, -100.0, 0.0])
    # cosine = np.dot(A, B) / (norm(A) * norm(B))
    # print(cosine(A,B))
    print(abs(B))
