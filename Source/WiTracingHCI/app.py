### Implementation

## We use T265 Tracking Camera to Produce Ground Truth Positiong of the Physical Twin.
## The camera also attach with a BLE scanner to collect the RSSI measurement.
## The combination of these two measurement will be treated as the measurement from real world to validate our digital twin.

import os
import time
import pyrealsense2 as rs

import settings
from device.t265 import T265Proxy
from device.ble import BLEProxy
from utils import CSVProxy

def millisec():
    return time.time_ns() // 1_000_000

if __name__ == "__main__":
    print("[INF] Begin.")

    BLE_PORT = settings.HARDWARE_CONFIG['ble_port']
    BLE_BAUDRATE = 96000

    csv_proxy = CSVProxy(wait_time=0.01)
    csv_proxy.start()

    def on_motion_recv(data):
        csv_proxy.enqueue_motion(data)

    def on_signal_recv(data):
        csv_proxy.enqueue_signal(data)


    t265_proxy = T265Proxy(on_data_recv_fn=on_motion_recv)
    t265_proxy.start()

    ble_proxy = BLEProxy(port=BLE_PORT, baudrate=BLE_BAUDRATE, on_data_recv_fn=on_signal_recv)
    ble_proxy.start()


    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

    ble_proxy.release()
    print("[INF] Completed!")