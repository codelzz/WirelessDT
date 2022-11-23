import socket
import time
import json
import settings
from networking.udp_socket_client import UdpSocketClient
from preprocess.preprocessor import Preprocessor
from dl.predictor import Predictor

if __name__ == "__main__":
    # SERVER_ENDPOINT = settings.NETWORK_CONFIG['server_endpoint']
    CLIENT_ENDPOINT = settings.NETWORK_CONFIG['client_endpoint']

    preprocessor = None
    predictor = None
    client = None
    
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
        data = json.loads(byte_data.decode('utf-8'))
        data = {k.lower(): v for k, v in data.items()} # convert to lower
        # ['txname','rxname','coordinates','rssi','timestamp']
        # print(data)
        preprocessor.enqueue(data)
        # predictor.enqueue(data)

    preprocessor = Preprocessor(wait_time=0.01)
    preprocessor.start()

    #predictor = Predictor(on_predict=on_predict)
    #predictor.start()

    client = UdpSocketClient(CLIENT_ENDPOINT, on_data_sent=on_data_sent, on_data_recv=on_data_recv)
    client.start()

    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break