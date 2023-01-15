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

        self.info = dict()

        def on_data_sent(byte_data, address):
            print(f"{address} << {repr(byte_data)}")

        def on_data_recv(byte_data, address):
            # print(f"{address} >> {repr(byte_data)}")
            temp = byte_data.decode('utf8').replace("'", '"')
            if '}' in temp:
                # print(temp)
                received_data_json = json.loads(temp)
                self.tx_readings[received_data_json["txname"]] = received_data_json["rssi"]
                self.is_receiving = True
                # print(self.tx_readings)

        self.udp_server = UdpSocketClient(self.CLIENT_ENDPOINT, on_data_sent=on_data_sent, on_data_recv=on_data_recv)
        self.udp_server.start()

    def _get_info(self):
        # json_object = json.loads(self.received_data)
        return self.info

    def _get_obs(self):
        if self.is_receiving:
            txs = np.fromiter(self.tx_readings.values(), dtype=int)
            return {
                "TXs": txs,
            }
        else:
            return {
                "TXs": np.array([-255, -255, -255, -255, -255, -255, -255, -255, -255, ]),
            }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        data = self.reset_message
        data = json.dumps(data)
        byte_data = str.encode(data)
        self.udp_server.sendto(byte_data=byte_data, address=self.SERVER_ENDPOINT)
        print("Reseting Env ...")
        time.sleep(1 / 5)

        obs = {
            "TXs": np.array([-255, -255, -255, -255, -255, -255, -255, -255, -255, ]),
        }
        info = dict()

        return obs, info

    def step(self, action):
        observation = self._get_obs()
        reward = 1
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        pass

    def close(self):
        pass
