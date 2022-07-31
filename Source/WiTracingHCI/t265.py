import os
import time

import settings

from device.t265 import T265Proxy

if __name__ == "__main__":
    print("[INF] Begin.")

    # T265 -----------------------------------
    def on_motion_recv(data):
        print(f"x: {data['x']:.2f} y: {data['y']:.2f} z: {data['z']:.2f} pitch: {data['pitch']:.2f} yaw: {data['yaw']:.2f} roll: {data['roll']:.2f}")
    t265_proxy = T265Proxy(on_data_recv_fn=on_motion_recv)
    t265_proxy.start()

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

    print("[INF] Completed!")