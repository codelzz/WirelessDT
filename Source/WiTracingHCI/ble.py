import os
import time

import settings

from device.ble import BLEProxy

if __name__ == "__main__":
    print("[INF] Begin.")

    # BLE
    BLE_PORT = settings.HARDWARE_CONFIG['ble_port']
    BLE_BAUDRATE = 96000

    # BLE ------------------------------------
    def on_signal_recv(data):
        print(f"{data['address']} rssi:{data['rssi']}")

    ble_proxies = []

    for port in BLE_PORT:
        ble_proxies.append(BLEProxy(port=port, baudrate=BLE_BAUDRATE, on_data_recv_fn=on_signal_recv))
    
    for proxy in ble_proxies:
        proxy.start()

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

    for proxy in ble_proxies:
        proxy.release()
    print("[INF] Completed!")