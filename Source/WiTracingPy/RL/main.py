import gymnasium as gym
import RL

import socket
import time

import settings

import json
import random

from agent.rltrack_agent import Rltrack_Agent
from networking.udp_socket_client import UdpSocketClient

forward_message = {"move_forward": True,
                   "turn_left": False,
                   "turn_right": False,
                   "reset": False,
                   }
turnright_message = {"move_forward": False,
                     "turn_left": False,
                     "turn_right": True,
                     "reset": False,
                     }
turnleft_message = {"move_forward": False,
                    "turn_left": True,
                    "turn_right": False,
                    "reset": False,
                    }
stop_message = {"move_forward": False,
                "turn_left": False,
                "turn_right": False,
                "reset": False,
                }
reset_message = {"move_forward": False,
                 "turn_left": False,
                 "turn_right": False,
                 "reset": True,
                 }


if __name__ == "__main__":
    env = gym.make('RL/RLTrack-v0')
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
    total_num_episodes = int(5e3)
    obs_space_dims = env.observation_space["TXs"].shape[0]
    action_space_dims = env.action_space.n
    print(obs_space_dims)
    print(action_space_dims)

    agent = Rltrack_Agent(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset()

        done = False
        # while not done:
        for steps in range(50):
            action = agent.sample_action(obs)

            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)
            time.sleep(1/5)
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


