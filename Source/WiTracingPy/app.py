import socket
import time

from networking.udp_client import UdpClient

from networking.udp_client_unreal import UdpClientUnreal
import settings


if __name__ == "__main__":
    SERVER_ENDPOINT = settings.NETWORK_CONFIG['server_endpoint']
    CLIENT_ENDPOINT = settings.NETWORK_CONFIG['client_endpoint']

    def on_data_sent(byte_data, address):
        print(f"{address} << {repr(byte_data)}")

    def on_data_recv(byte_data, address):
        print(f"{address} >> {repr(byte_data)}")

    client = UdpClientUnreal(server_endpoint=SERVER_ENDPOINT,
                             client_endpoint=CLIENT_ENDPOINT,
                             on_data_sent=on_data_sent,
                             on_data_recv=on_data_recv)
    client.start()

    while True:
        data = f'{time.time()}'
        byte_data = str.encode(data)
        client.send(byte_data=byte_data)
        time.sleep(1)
