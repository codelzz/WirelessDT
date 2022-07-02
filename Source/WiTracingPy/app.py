import socket
import time

import settings
from networking.udp_socket_client import UdpSocketClient
from networking.unreal_udp_socket_client import UnrealUdpSocketClient


if __name__ == "__main__":
    SERVER_ENDPOINT = settings.NETWORK_CONFIG['server_endpoint']
    CLIENT_ENDPOINT = settings.NETWORK_CONFIG['client_endpoint']

    def on_data_sent(byte_data, address):
        print(f"{address} << {repr(byte_data)}")

    def on_data_recv(byte_data, address):
        print(f"{address} >> {repr(byte_data)}")

    client = UdpSocketClient(CLIENT_ENDPOINT, on_data_sent=on_data_sent, on_data_recv=on_data_recv)
    client.start()

    while True:
        # [ISSUE] why increasing decimal places can help to eliminate the 
        # error buffer at unreal server side?
        data = f'{time.time():.10f}'
        byte_data = data.encode("utf-8")
        client.sendto(byte_data=byte_data, address=SERVER_ENDPOINT)
        time.sleep(0.1)


