import serial
import time
import pyrealsense2 as rs

from thread.runnable import Runnable
import utils


# Socket receiver handle data receiving task
class BLEProxy(Runnable):
    """
    BLE Proxy - Communicate with Arduino board for ble related data acquisition
    """
    def __init__(self, port, baudrate, on_data_recv_fn, address_filter=None, wait_time=0.001):
        super(BLEProxy, self).__init__(wait_time=wait_time)
        self.port = port
        self.baudrate = baudrate
        self.on_data_recv_fn = on_data_recv_fn
        self.serial_timeout = .1
        self.serial = serial.Serial(port=self.port, baudrate=self.baudrate , timeout=self.serial_timeout)
        self.payload = {}
        self.address_filter = address_filter
        self.print(f'Connected to {self.port}')

    def do(self):
        try:
            data = self.recv_data()
            if len(data) > 0 and self.on_data_recv_fn is not None:
                data = self.parse_data(data)
                if self.address_filter is not None:
                    if data[0] in self.address_filter.keys():
                        self.update_payload(data)
                        self.on_data_recv_fn(self.payload)
                else:
                    self.update_payload(data)
                    self.on_data_recv_fn(self.payload)
        except serial.serialutil.SerialException as e:
            # exception happen when board is restarted while connecting
            self.print(f'Connection lost')
            time.sleep(1)
            try:
                self.print(f'Attemp to reconnect {self.port}')
                self.serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.serial_timeout)
                self.print(f'Connected to {self.port}')
            except serial.serialutil.SerialException as e:
                # exception happen when reconnecting the non-standby board (not ready for connection)
                time.sleep(1)

    def recv_data(self):
        self.serial.write(bytes('', 'utf-8'))
        return self.serial.readline()

    def parse_data(self, data):
        return data.decode("utf-8")[:-2].split(',')

    def release(self):
        self.serial.close()
        self.print(f'Connection closed')

    def update_payload(self, data):
        self.payload = {
            'timestamp':utils.millisecond(),
            'tx':data[0],
            'rssi':data[1],
        }