import time
import csv
import queue
import time
import settings
import os

from thread.runnable import Runnable


def millisecond():
    return time.time_ns() // 1_000_000

# csv
MOTION_DATA_CSV = settings.DATA_CONFIG['motion']
SIGNAL_DATA_CSV = settings.DATA_CONFIG['signal']
# header
MOTION_DATA_FIELDS = settings.DATA_CONFIG['motion_fields']
SIGNAL_DATA_FIELDS = settings.DATA_CONFIG['signal_fields']

class CSVProxy(Runnable):
    def __init__(self, wait_time=0.01):
        super(CSVProxy, self).__init__(wait_time=wait_time)
        self.motion_data_csv = MOTION_DATA_CSV
        self.signal_data_csv = SIGNAL_DATA_CSV
        # data queue
        self.motion_data_queue = queue.Queue()
        self.signal_data_queue = queue.Queue()
        # csv header
        self.motion_data_fields = MOTION_DATA_FIELDS
        self.signal_data_fields = SIGNAL_DATA_FIELDS
        # prepare
        self.prepare()
        self.print("Ready.")

    def prepare(self):
        self.init_csv(csv_file=self.motion_data_csv, fields=self.motion_data_fields)
        self.init_csv(csv_file=self.signal_data_csv, fields=self.signal_data_fields)

    @staticmethod 
    def init_csv(csv_file, fields):
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                f.close()

    @staticmethod
    def write_csv(csv_file, fields, row):
        with open(csv_file, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writerow(row)
            f.close()

    def format_motion_data_to_row(self, data):
        row = {}
        for field in MOTION_DATA_FIELDS:
            if field in data.keys():
                row[field] = data[field]
        return row

    def format_signal_data_to_row(self, data):
        row = {}
        for field in SIGNAL_DATA_FIELDS:
            if field in data.keys():
                row[field] = data[field]
        return row

    def do(self):
        if not self.motion_data_queue.empty():
            data = self.motion_data_queue.get()
            self.save(csv_file=self.motion_data_csv, fields=self.motion_data_fields, row=self.format_motion_data_to_row(data))
        if not self.signal_data_queue.empty():
            data = self.signal_data_queue.get()
            self.save(csv_file=self.signal_data_csv, fields=self.signal_data_fields, row=self.format_signal_data_to_row(data))

    def enqueue_motion(self, data):
        self.motion_data_queue.put(data)

    def enqueue_signal(self, data):
        self.signal_data_queue.put(data)

    def save(self, csv_file, fields, row):
        self.write_csv(csv_file=csv_file, fields=fields, row=row)
