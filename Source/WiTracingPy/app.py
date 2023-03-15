import re
import socket
import time
import json
import settings
from networking.udp_socket_client import UdpSocketClient
from networking.tcp_socket_server import TcpSocketServer

from preprocess.preprocessor import Preprocessor
from dl.predictor import Predictor

if __name__ == "__main__":
    # SERVER_ENDPOINT = settings.NETWORK_CONFIG['server_endpoint']
    # CLIENT_ENDPOINT = settings.NETWORK_CONFIG['client_endpoint']

    preprocessor = None
    predictor = None
    client = None
    server = None
    
    def on_predict(result):
        print(result)
        if len(result) == 3:
            result = result.numpy()
            data = {'x': float(result[0]), 'y': float(result[1]), 'z': float(result[2])}
            json_obj = json.dumps(data, indent = 0)
            byte_data = str.encode(str(json_obj),'utf-8')
            client.sendto(byte_data, ("127.0.0.1", 7000))  

    def on_data_sent(byte_data, address):
        print(f"{address} << {repr(byte_data)}")

    def on_data_recv(byte_data, address):
        # print(f"{address} >> {repr(byte_data)}")
        str_data = byte_data.decode('utf-8')
        try:
            str_list = re.split('({[^}]*})', str_data)[1::2]
            for s in str_list:
                data = json.loads(s)
                data = {k.lower(): v for k, v in data.items()} # convert to lower
                preprocessor.enqueue(data)
        except:
            print(f'Fail to parse:\n {str_data}\n')

    preprocessor = Preprocessor(wait_time=2)
    preprocessor.start()

    #predictor = Predictor(on_predict=on_predict)
    #predictor.start()

    # client = UdpSocketClient(CLIENT_ENDPOINT, on_data_sent=on_data_sent, on_data_recv=on_data_recv)
    # client.start()

    server = TcpSocketServer(endpoint=("192.168.31.200", 7777), on_data_recv=on_data_recv)
    server.start()

    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break