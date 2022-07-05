import socket
import time
import json
import settings
from networking.udp_socket_client import UdpSocketClient
from preprocess.preprocessor import Preprocessor

if __name__ == "__main__":
    SERVER_ENDPOINT = settings.NETWORK_CONFIG['server_endpoint']
    CLIENT_ENDPOINT = settings.NETWORK_CONFIG['client_endpoint']

    preprocessor = Preprocessor(wait_time=0.01)
    preprocessor.start()

    def on_data_sent(byte_data, address):
        print(f"{address} << {repr(byte_data)}")

    def on_data_recv(byte_data, address):
        print(f"{address} >> {repr(byte_data)}")
        data = json.loads(byte_data.decode('utf-8'))
        preprocessor.enqueue(data)

    client = UdpSocketClient(CLIENT_ENDPOINT, on_data_sent=on_data_sent, on_data_recv=on_data_recv)
    client.start()

    while True:
        # [ISSUE] why increasing decimal places can help to eliminate the 
        # error buffer at unreal server side?
        # data = f'{time.time():.10f}'
        # byte_data = data.encode("utf-8")
        # client.sendto(byte_data=byte_data, address=SERVER_ENDPOINT)
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break