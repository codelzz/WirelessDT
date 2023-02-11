import csv
import queue
import time
import settings
import os

from udpthread.runnable import Runnable

CSV_FILE = settings.APP_CONFIG['preprocessed_csv_file']
RSSI_MAX = settings.UNREAL_CONFIG['rssi_max']
RSSI_MIN = settings.UNREAL_CONFIG['rssi_min']
CSV_FIELDS = ['tx','x','y','z','rssi','timestamp'] 

class Preprocessor(Runnable):
    def __init__(self, wait_time=0.01):
        super(Preprocessor, self).__init__(wait_time=wait_time)
        self.data_queue = queue.Queue()
        self.csv_fields = CSV_FIELDS
        self.init_csv()
        self.print("Ready.")

    # csv file operation
    def init_csv(self):
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields)
                w.writeheader()
                f.close()

    def write_csv(self, data):
        with open(CSV_FILE, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.csv_fields)
            w.writerow(self.convert_to_csv_row(data))
            f.close()

    def convert_to_csv_row(self, data):
        csv_row = {'tx': data['txname'],
        		   'x': data['rxx'],
                   'y': data['rxy'],
                   'z': data['rxz'],
                   'rssi': max(data['rssi'], -255),
                   'timestamp': data['timestamp']}
        return csv_row
	# csv file operation end

    def do(self):
        if not self.data_queue.empty():
            # data = self.data_queue.get()
            # self.preprocess(data)
            with open(CSV_FILE, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields)
                while not self.data_queue.empty():
                    w.writerow(self.convert_to_csv_row(self.data_queue.get()))
                f.close()

    def enqueue(self, data):
        self.data_queue.put(data)

    def preprocess(self, data):
        self.write_csv(data)