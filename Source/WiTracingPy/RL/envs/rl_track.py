import gymnasium as gym
import numpy as np
from gymnasium import spaces

import socket
import time

import settings
from networking.udp import UdpSocketReceiver, UdpSocketSender
from networking.udp_socket_client import UdpSocketClient

import json
import random


class RLtrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

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

    SERVER_ENDPOINT = settings.NETWORK_CONFIG['server_endpoint']
    CLIENT_ENDPOINT = settings.NETWORK_CONFIG['client_endpoint']

    action_list = [forward_message, turnright_message, turnleft_message, stop_message]
    # action_list = ["forward", "turnright", "turnleft", "stop"]

    def __init__(self, render_mode=None, Tx_num=9):
        self.observation_space = spaces.Dict(
            {
                "TXs": spaces.Box(-255, 0, shape=(Tx_num,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.is_receiving = False
        self.tx_readings = dict()

        self.reward = 0

        # self.info = dict()

        def on_data_sent(byte_data, address):
            # print(f"{address} << {repr(byte_data)}")
            pass

        def on_data_recv(byte_data, address):
            # print(f"{address} >> {repr(byte_data)}")
            temp = byte_data.decode('utf8').replace("'", '"')
            if 'rssi' in temp:
                # print(temp)
                received_data_json = json.loads(temp)
                self.tx_readings[received_data_json["txname"]] = received_data_json["rssi"]

                self.is_receiving = True
                # print(self.tx_readings)
            if 'reward' in temp:
                # print(f"{address} >> {repr(temp)}")
                received_data_json = json.loads(temp)
                self.reward = -received_data_json["position_reward"]

        self.udp_server = UdpSocketClient(self.CLIENT_ENDPOINT, on_data_sent=on_data_sent, on_data_recv=on_data_recv)
        self.udp_server.start()

    def _get_info(self):
        # json_object = json.loads(self.received_data)
        return dict()

    def _get_obs(self):
        if self.is_receiving:
            txs = np.fromiter(self.tx_readings.values(), dtype=int)
            # print(txs)
            return {
                "TXs": txs,
            }
        else:
            return {
                "TXs": np.array([-255, -255, -255, -255, -255, -255, -255, -255, -255]),
            }

    def reset(self, seed=None, options=None):
        print("Reseting Env ...")
        super().reset(seed=seed)
        data = self.reset_message
        data = json.dumps(data)
        byte_data = str.encode(data)
        self.udp_server.sendto(byte_data=byte_data, address=self.SERVER_ENDPOINT)

        obs = {
            "TXs": np.array([-255, -255, -255, -255, -255, -255, -255, -255, -255]),
        }
        self.reward = 0
        info = self._get_info()

        print("Receiving TXs data from engine ...")
        time.sleep(1)

        return obs, info

    def step(self, action):
        action = (action - action.min()) / (action.max() - action.min())

        # Send action to Engine
        message = random.choices(self.action_list, weights=action)[0]
        message = json.dumps(message)
        byte_message = str.encode(message)
        self.udp_server.sendto(byte_data=byte_message, address=self.SERVER_ENDPOINT)

        observation = self._get_obs()
        reward = self.reward
        # print(reward)
        terminated = False

        if reward < -100:
            truncated = True
        else:
            truncated = False

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
