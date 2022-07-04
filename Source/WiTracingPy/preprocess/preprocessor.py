import csv
import queue
import time
import settings
import os


from thread.runnable import Runnable

CSV_FILE = settings.APP_CONFIG['preprocessed_csv_file']
RSSI_MAX = settings.UNREAL_CONFIG['rssi_max']
RSSI_MIN = settings.UNREAL_CONFIG['rssi_min']

class Preprocessor(Runnable):
    def __init__(self, wait_time=0.01):
        super(Preprocessor, self).__init__(wait_time=wait_time)
        self.data_queue = queue.Queue()
        self.csv_fields = ['tag', 'x', 'y', 'z', 'timestamp']+[ f'{x}' for x in range(RSSI_MAX, RSSI_MIN-1,-1)]
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
        csv_row = {'tag': data['tag'],
        		   'x': data['coordinates'][0],
                   'y': data['coordinates'][1],
                   'z': data['coordinates'][2],
                   'timestamp': data['timestamp']}
        for i, x in enumerate(range(RSSI_MAX, RSSI_MIN-1,-1)):
            key = f'{x}'
            csv_row[key] = data['rssipdf'][i]
        return csv_row
	# csv file operation end

    def do(self):
        if not self.data_queue.empty():
            data = self.data_queue.get()
            self.preprocess(data)

    def enqueue(self, data):
        self.data_queue.put(data)

    def preprocess(self, data):
        self.write_csv(data)