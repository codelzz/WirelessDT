### Implementation

## We use T265 Tracking Camera to Produce Ground Truth Positiong of the Physical Twin.
## The camera also attach with a BLE scanner to collect the RSSI measurement.
## The combination of these two measurement will be treated as the measurement from real world to validate our digital twin.

import os
import time
import json

import settings
from network.udp_socket import UdpSocketClient
from device.t265 import T265Proxy
from device.ble import BLEProxy
from utils import CSVProxy


if __name__ == "__main__":
    print("[INF] Begin.")

    # Server
    SERVER_ENDPOINT = settings.APP_CONFIG['server_endpoint']
    CLIENT_ENDPOINT = settings.APP_CONFIG['client_endpoint']
    # BLE
    BLE_PORT = settings.HARDWARE_CONFIG['ble_port']
    BLE_BAUDRATE = 96000

    csv_proxy = None
    client = None
    t265_proxy = None
    ble_proxy = None

    # CSV -----------------------------------
    csv_proxy = CSVProxy(wait_time=0.01)

    # Client -----------------------------------
    def on_data_sent(byte_data, address):
        data = json.loads(byte_data.decode('utf-8'))
        if bool(data):
            print(f"x: {data['x']:.2f} y: {data['y']:.2f} z: {data['z']:.2f} " +
                  f"pitch: {data['pitch']:.2f} yaw: {data['yaw']:.2f} roll: {data['roll']:.2f} " +
                  f"address: {data['address']} rssi: {data['rssi']}")
        # print(f"{address} << {repr(byte_data)}")

    def on_data_recv(byte_data, address):
        print(f"{address} >> {repr(byte_data)}")

    client = UdpSocketClient(CLIENT_ENDPOINT, on_data_sent=on_data_sent, on_data_recv=on_data_recv)

    # Helper
    def merge_data(motion, signal):
        data = {}
        if bool(motion) and bool(signal):
            data = {
                    'x':motion['x'],
                    'y':motion['y'],
                    'z':motion['z'],
                    'pitch':motion['pitch'],
                    'yaw':motion['yaw'],
                    'roll':motion['roll'],
                    'address':"n/a",
                    'rssi':-255,
                    }
            # if there is a present measurement from BLE
            # instantaneity = motion['timestamp'] - signal['timestamp']
            # if instantaneity > 0 and instantaneity < 2.5: # ms
            data['address'] = signal['address']
            data['rssi'] = signal['rssi']
        return data

    # T265 -----------------------------------
    def on_motion_recv(data):
        csv_proxy.enqueue_motion(data)
        client.sendjson(merge_data(motion=t265_proxy.payload, signal=ble_proxy.payload), SERVER_ENDPOINT)
    t265_proxy = T265Proxy(on_data_recv_fn=on_motion_recv)
    
    # BLE ------------------------------------
    def on_signal_recv(data):
        csv_proxy.enqueue_signal(data)
        client.sendjson(merge_data(motion=t265_proxy.payload, signal=ble_proxy.payload), SERVER_ENDPOINT)
    ble_proxy = BLEProxy(port=BLE_PORT, baudrate=BLE_BAUDRATE, on_data_recv_fn=on_signal_recv)
   
    csv_proxy.start()
    client.start()
    t265_proxy.start()
    ble_proxy.start()

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

    ble_proxy.release()
    print("[INF] Completed!")