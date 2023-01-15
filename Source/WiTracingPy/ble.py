# rssi_collector
import os
import serial
import time
import queue
import csv
import argparse

from udpthread.runnable import Runnable

# BLE proxy
BAUDRATE = 96000

CSV_FILE = "data/real_ble_data"

# // ----------------------------------------------------------------------------------------------------------------------------
# // |  TXAddress | Packet Index | RSSI | Count | RX Speed | TX Speed | Last Time Elapsed | Total Time Elasped | Timestamp (ms) |
# // ----------------------------------------------------------------------------------------------------------------------------
CSV_FIELD = ['address','packet_id','rssi','count','rx_speed','tx_speed','last_time_elapsed','total_time_elapsed','timestamp']

# Socket receiver handle data receiving task
class BLEProxy(Runnable):
    """
    BLE Proxy - Communicate with Arduino board for ble related data acquisition
    """
    def __init__(self, port, baudrate, on_data_recv_fn, wait_time=0.001):
        super(BLEProxy, self).__init__(wait_time=wait_time)
        self.port = port
        self.baudrate = baudrate
        self.on_data_recv_fn = on_data_recv_fn
        self.serial_timeout = .1
        self.serial = serial.Serial(port=self.port, baudrate=self.baudrate , timeout=self.serial_timeout)
        self.print("Ready.")

    def do(self):
        try:
            data = self.recv_data()
            if len(data) > 0 and self.on_data_recv_fn is not None:
                self.on_data_recv_fn(data)
        except serial.serialutil.SerialException as e:
        	# exception happen when board is restarted while connecting
            time.sleep(1)
            try:
                self.serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.serial_timeout)
            except serial.serialutil.SerialException as e:
            	# exception happen when reconnecting the non-standby board (not ready for connection)
                time.sleep(1)

    def recv_data(self):
        self.serial.write(bytes('', 'utf-8'))
        return self.serial.readline()


class CSVProxy(Runnable):
    def __init__(self, csv_file, wait_time=0.01):
        super(CSVProxy, self).__init__(wait_time=wait_time)
        self.data_queue = queue.Queue()
        self.csv_fields = CSV_FIELD
        self.csv_file = csv_file
        self.init_csv()
        self.print("Ready.")

    # csv file operation
    def init_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields)
                w.writeheader()
                f.close()

    def write_csv(self, data):
        with open(self.csv_file, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.csv_fields)
            csv_row, ok = self.convert_to_csv_row(data)
            if ok:
                w.writerow(csv_row)
            f.close()

    def convert_to_csv_row(self, data):
        csv_row = {}
        split_data = data.split(',')

        if len(split_data) != len(self.csv_fields):
            return csv_row, False
        for i, col in enumerate(self.csv_fields):
            csv_row[col] = split_data[i]
        return csv_row, True
    # csv file operation end

    def do(self):
        if not self.data_queue.empty():
            data = self.data_queue.get()
            self.write_csv(data)

    def enqueue(self, data):
        self.data_queue.put(data)


def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', dest='port', type=str, help='set the port for device serial')
    args = parser.parse_args()
    # args check
    assert args.port is not None, "--port should be specified"
    return args

if __name__ == "__main__":
    args = parse_argv()
    csv_file = CSV_FILE + "_" + args.port + ".csv"

    csv_proxy = CSVProxy(csv_file=csv_file, wait_time=0.01)
    csv_proxy.start()

    def on_data_recv(data):
        str_data = data.decode("utf-8")[:-2] # [:-2] remove "\r\n"
        csv_proxy.enqueue(str_data)
        print(str_data)

    ble_proxy = BLEProxy(port=args.port, baudrate=BAUDRATE, on_data_recv_fn = on_data_recv)
    ble_proxy.start()

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break