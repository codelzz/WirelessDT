import socket
import time
import json
import settings
from networking.udp_socket_client import UdpSocketClient

if __name__ == "__main__":
    # SERVER_ENDPOINT = settings.NETWORK_CONFIG['server_endpoint']
    CLIENT_ENDPOINT = ("192.168.31.110", 8888)

    client = None


    def on_data_sent(byte_data, address):
        print(f"{address} << {repr(byte_data)}")

    def on_data_recv(byte_data, address):
        print(f"{address} >> {repr(byte_data)}")
        # data = json.loads(byte_data.decode('utf-8'))
        # data = {k.lower(): v for k, v in data.items()} # convert to lower
        # print(data)

    client = UdpSocketClient(CLIENT_ENDPOINT, on_data_sent=on_data_sent, on_data_recv=on_data_recv)
    print('Start')
    client.start()

    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break