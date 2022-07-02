import socket
import time

from networking import udp
from networking.udp import UdpSocketRunnable


class UdpSocketServer(UdpSocketRunnable):
    endpoint = None

    def __init__(self, endpoint, wait_time=0.01):
        super().__init__(wait_time=wait_time)
        self.endpoint = endpoint
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.socket.bind(self.endpoint)

    # ~ end Runnable interface
    def do(self):
        packet = self.socket.recvfrom(self. max_buffer_size)
        data, sender = packet
        self.print(f"{udp.endpoint_to_string(sender)} >> {repr(data)}")
        self.socket.sendto(data, sender)
        self.print(f"{udp.endpoint_to_string(sender)} << {repr(data)}")

    def __repr__(self):
        return f"<{self.__class__.__name__} IP={self.endpoint[0]} Port={self.endpoint[1]}>"


# if __name__ == "__main__":
#     server = UdpScoketServer(settings.NETWORK_CONFIG['server_endpoint'])
#     server.start()

#     while True:
#         time.sleep(60)
