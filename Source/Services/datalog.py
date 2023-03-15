
# datalog is a logger service saving data to local repo during simulation

import os
import csv
import asyncio
import queue
from service import Service
import settings

class WiTracingLog:
    raw_file = settings.DATALOG_CONFIG['raw_file']
    rssi_max = settings.DATALOG_CONFIG['rssi_max']
    rssi_min = settings.DATALOG_CONFIG['rssi_min']
    csv_fields = ['tx','x','y','z','rssi','timestamp'] 

    def __init__(self):
        self.data_queue = queue.Queue()
        self.init_csv()

    # csv file operation
    def init_csv(self):
        if not os.path.exists(self.raw_file):
            with open(self.raw_file, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields)
                w.writeheader()
                f.close()

    def convert_to_csv_row(self, data):
        csv_row = {'tx': data['txname'],
                   'x': data['rxx'],
                   'y': data['rxy'],
                   'z': data['rxz'],
                   'rssi': max(data['rssi'], -255),
                   'timestamp': data['timestamp']}
        return csv_row

    def do(self):
        if not self.data_queue.empty():
            # data = self.data_queue.get()
            # self.preprocess(data)
            with open(self.raw_file, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields)
                while not self.data_queue.empty():
                    w.writerow(self.convert_to_csv_row(self.data_queue.get()))
                f.close()

    def enqueue(self, data):
        self.data_queue.put(data)

class DataLogService(Service):
    def __init__(self):
        super(DataLogService).__init__()
        self.tasks = [WiTracingLog()]

    # core async function
    async def do(self):
    	# log data every 1 seconds
        await asyncio.sleep(1)
        for task in self.tasks:
            task.do()

    def enqueue(self, data):
        for task in self.tasks:
            if isinstance(task, WiTracingLog):
                task.enqueue(data)


if __name__ == "__main__":
    datalog = DataLogService()
    datalog.run()