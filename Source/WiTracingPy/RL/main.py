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

if __name__ == "__main__":
    env = gym.make('RL/RLTrack-v0')
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
    total_num_episodes = int(5e2)
    obs_space_dims = env.observation_space["TXs"].shape[0]
    action_space_dims = env.action_space.shape[0]
    print(obs_space_dims)
    print(action_space_dims)

    rewards_over_seeds = []

    for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        agent = Rltrack_Agent(obs_space_dims, action_space_dims)
        reward_over_episodes = []

        print(agent.net)

        for episode in range(total_num_episodes):
            # gymnasium v26 requires users to set seed while resetting the environment
            obs, info = wrapped_env.reset()

            done = False
            while not done:
                # for steps in range(50):
                act = agent.sample_action(obs)
                action = env.action_space.sample()

                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                agent.rewards.append(reward)
                time.sleep(1 / 5)

                done = terminated or truncated

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            agent.update()

            if episode % 10 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print("Episode:", episode, "Average Reward:", avg_reward)

        rewards_over_seeds.append(reward_over_episodes)

        rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
        df1 = pd.DataFrame(rewards_to_plot).melt()
        df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
        sns.set(style="darkgrid", context="talk", palette="rainbow")
        sns.lineplot(x="episodes", y="reward", data=df1).set(
            title="REINFORCE for RLTrack-v0"
        )
        plt.show()

    #
    #     # action_list = [forward_message, turnright_message, turnleft_message, stop_message]
    #     # env.reset()
    #     # # for _ in range(50):
    #     # while True:
    #     #     observation, reward, terminated, _, info = env.step(1)
    #     #     print(observation)
    #     #     env.render()
    #     #     time.sleep(1 / 5)
    #
    env.close()
