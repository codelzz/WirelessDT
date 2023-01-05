import socket
import time

import settings
from networking.udp import UdpSocketReceiver, UdpSocketSender

import json
import random

import gymnasium as gym

class UdpSocketClient:

    def __init__(self, endpoint, on_data_sent, on_data_recv, wait_time=0.001):
        self.endpoint = endpoint
        self.on_data_sent = on_data_sent
        self.on_data_recv = on_data_recv
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.socket.bind(self.endpoint)
        self.receiver = UdpSocketReceiver(self.socket, self.on_data_recv, wait_time=wait_time)
        self.sender = UdpSocketSender(self.socket, self.on_data_sent, wait_time=wait_time)

    def start(self):
        if self.sender is not None:
            self.sender.start()

        if self.receiver is not None:
            self.receiver.start()

    def sendto(self, byte_data, address):
        self.sender.sendto(byte_data=byte_data, address=address)


if __name__ == "__main__":
    SERVER_ENDPOINT = settings.NETWORK_CONFIG['server_endpoint']
    CLIENT_ENDPOINT = settings.NETWORK_CONFIG['client_endpoint']


    def on_data_sent(byte_data, address):
        print(f"{address} << {repr(byte_data)}")


    def on_data_recv(byte_data, address):
        print(f"{address} >> {repr(byte_data)}")

    client = UdpSocketClient(CLIENT_ENDPOINT, on_data_sent=on_data_sent, on_data_recv=on_data_recv)
    client.start()

    # action_list = ['forward_message', 'turn_left', 'turn_right', 'stop']

    forward_message = {"move_forward": True,
                       "turn_left": False,
                       "turn_right": False,
                       }
    turnright_message = {"move_forward": False,
                         "turn_left": False,
                         "turn_right": True,
                         }
    turnleft_message = {"move_forward": False,
                        "turn_left": True,
                        "turn_right": False,
                        }
    stop_message = {"move_forward": False,
                    "turn_left": False,
                    "turn_right": False,
                    }

    action_list = [forward_message, turnright_message, turnleft_message, stop_message]

    while True:
        data = random.choice(action_list)
        # data = stop_message
        data = json.dumps(data)
        byte_data = str.encode(data)
        client.sendto(byte_data=byte_data, address=SERVER_ENDPOINT)
        time.sleep(1/5)

    # while True:
    #     # data = f'{time.time()}'
    #     data = action
    #     byte_data = str.encode(data)
    #     client.sendto(byte_data=byte_data, address=SERVER_ENDPOINT)
    #     time.sleep(1)
