import socket
import time

import settings
from networking.udp import UdpReceiver, UdpSender


class UdpClient:
    endpoint = None
    on_data_sent = None
    on_data_recv = None
    receiver = None
    sender = None
    socket = None

    def __init__(self, endpoint, on_data_sent, on_data_recv):
        self.endpoint = endpoint
        self.on_data_recv = on_data_recv
        self.on_data_sent = on_data_sent
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.socket.bind(self.endpoint)
        self.receiver = UdpReceiver(self.socket, self.on_data_recv, wait_time=0.001)
        self.sender = UdpSender(self.socket, self.on_data_sent, wait_time=0.001)

    def start(self):
        if self.sender is not None:
            self.sender.start()

        if self.receiver is not None:
            self.receiver.start()

    def sendto(self, byte_data, address):
        self.sender.sendto(byte_data=byte_data, address=address)


if __name__ == "__main__":
    server_address = settings.NETWORK_CONFIG['server_endpoint']

    def on_data_sent(byte_data, address):
        print(f"{address} << {repr(byte_data)}")

    def on_data_recv(byte_data, address):
        print(f"{address} >> {repr(byte_data)}")

    client = UdpClient(server_address, on_data_sent=on_data_sent, on_data_recv=on_data_recv)
    client.start()

    while True:
        data = f'{time.time()}'
        byte_data = str.encode(data)
        client.sendto(byte_data=byte_data, address=server_address)
        time.sleep(1)




