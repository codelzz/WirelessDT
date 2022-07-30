import time

import settings
from network.udp_socket import UdpSocketClient

import json

if __name__ == "__main__":
    SERVER_ENDPOINT = settings.APP_CONFIG['server_endpoint']
    CLIENT_ENDPOINT = settings.APP_CONFIG['client_endpoint']

    def on_data_sent(byte_data, address):
        print(f"{address} << {repr(byte_data)}")

    def on_data_recv(byte_data, address):
        print(f"{address} >> {repr(byte_data)}")

    client = UdpSocketClient(CLIENT_ENDPOINT, on_data_sent=on_data_sent, on_data_recv=on_data_recv)
    client.start()

    x,y,z = 0,0,0
    while True:
        # data = f'{time.time()}'
        x += 1;
        y += 1;
        z += 0.1;
        packet={'x':x,'y':y,'z':z,'rssi':0}
        json_obj = json.dumps(packet, indent = 0) 
        byte_data = str.encode(str(json_obj),'utf-8')
        client.send(byte_data=byte_data, address=SERVER_ENDPOINT)
        time.sleep(0.01)
