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
from numpy.linalg import norm


class RLtrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}
    SERVER_ENDPOINT = settings.NETWORK_CONFIG['server_endpoint']
    CLIENT_ENDPOINT = settings.NETWORK_CONFIG['client_endpoint']

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

        self.last_rx_position = np.array([0.0, 0.0, 90.0])
        self.last_agent_position = np.array([0.0, 2500.0, 90.0])

        self.rx_position = np.array([0.0, 0.0, 90.0])
        self.agent_position = np.array([0.0, 2500.0, 90.0])

        self.last_txs = None

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
                # self.reward = -received_data_json["position_reward"]
                # print("rx_x: ", received_data_json["rx_x"])

                self.last_rx_position = self.rx_position
                self.last_agent_position = self.agent_position

                self.rx_position = np.array([received_data_json["rx_x"], received_data_json["rx_y"], received_data_json["rx_z"]])
                self.agent_position = np.array([received_data_json["agent_x"], received_data_json["agent_y"], received_data_json["agent_z"]])

        self.udp_server = UdpSocketClient(self.CLIENT_ENDPOINT, on_data_sent=on_data_sent, on_data_recv=on_data_recv)
        self.udp_server.start()

    def get_reward(self):
        rx_move = self.rx_position - self.last_rx_position
        agent_move = self.agent_position - self.last_agent_position

        # compute cosine similarity
        # cosine = np.dot(rx_move, agent_move) / (norm(rx_move) * norm(agent_move))
        # return cosine
        diff = rx_move - agent_move
        return -np.sum(abs(diff))


    def _get_info(self):
        # json_object = json.loads(self.received_data)
        return dict()

    def _get_obs(self):
        if self.is_receiving:
            txs = np.fromiter(self.tx_readings.values(), dtype=int)
            last_txs = np.copy(self.last_txs)
            self.last_txs = txs
            # print(txs)
            return {
                "Last_TXs": last_txs,
                "TXs": txs,
            }
        else:
            return {
                "Last_TXs": np.array([-255, -255, -255, -255, -255, -255, -255, -255, -255]),
                "TXs": np.array([-255, -255, -255, -255, -255, -255, -255, -255, -255]),
            }

    def reset(self, seed=None, options=None):
        print("Reseting Env ...")
        super().reset(seed=seed)
        reset_message = {"move_forward": False,
                          "turn_left": False,
                          "turn_right": False,
                          "reset": True,
                          }
        data = json.dumps(reset_message)
        byte_data = str.encode(data)
        self.udp_server.sendto(byte_data=byte_data, address=self.SERVER_ENDPOINT)

        obs = {
            "Last_TXs": np.array([-255, -255, -255, -255, -255, -255, -255, -255, -255]),
            "TXs": np.array([-255, -255, -255, -255, -255, -255, -255, -255, -255]),
        }
        self.last_txs = np.array([-255, -255, -255, -255, -255, -255, -255, -255, -255])
        self.reward = 0
        info = self._get_info()

        self.last_rx_position = np.array([0.0, 0.0, 90.0])
        self.last_agent_position = np.array([0.0, 2500.0, 90.0])

        self.rx_position = np.array([0.0, 0.0, 90.0])
        self.agent_position = np.array([0.0, 2500.0, 90.0])

        print("Receiving TXs data from engine ...")
        time.sleep(1)

        return obs, info

    def step(self, action):
        # action = (action - action.min()) / (action.max() - action.min())
        #
        # # Send action to Engine
        # message = random.choices(self.action_list, weights=action)[0]
        # message = json.dumps(message)
        # byte_message = str.encode(message)
        # self.udp_server.sendto(byte_data=byte_message, address=self.SERVER_ENDPOINT)
        action = action[0]
        vector_x = action.tolist()[0]
        vector_y = action.tolist()[1]
        action_message = {"move_forward": False,
                          "turn_left": False,
                          "turn_right": False,
                          "reset": False,
                          "vector_x": vector_x,
                          "vector_y": vector_y,
                          }
        action_message = json.dumps(action_message)
        byte_message = str.encode(action_message)
        self.udp_server.sendto(byte_data=byte_message, address=self.SERVER_ENDPOINT)
        # print("action sent!")
        observation = self._get_obs()
        reward = self.get_reward()
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
