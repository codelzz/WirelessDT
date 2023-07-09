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
    csv_fields = ['tx', 'x', 'y', 'z', 'rssi', 'timestamp']

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
        if 'txname' not in data:
            return None
        csv_row = {'tx': data['txname'],
                   'x': data['rxx'],
                   'y': data['rxy'],
                   'z': data['rxz'],
                   'rssi': max(data['rssi'], -255),
                   'timestamp': data['timestamp']}
        return csv_row

    def do(self):
        if not self.data_queue.empty():
            with open(self.raw_file, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields)
                while not self.data_queue.empty():
                    csv_row = self.convert_to_csv_row(self.data_queue.get())
                    if csv_row is not None:
                        w.writerow(csv_row)
                f.close()

    def enqueue(self, data):
        self.data_queue.put(data)


class CameraDetectionLog:
    raw_file = settings.CAM_DATALOG_CONFIG['raw_file']
    csv_fields = ['Ped_ID', 'x', 'y', 'world_x', 'world_y', 'los', 'timestamp']

    def __init__(self):
        self.data_queue = queue.Queue()
        self.init_csv()

    def init_csv(self):
        if not os.path.exists(self.raw_file):
            with open(self.raw_file, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields)
                w.writeheader()
                f.close()

    def convert_to_csv_row(self, data):
        if 'pedid' not in data:
            return None
        csv_row = {'Ped_ID': data['pedid'],
                   'x': data['camx'],
                   'y': data['camy'],
                   'world_x': data['worldx'],
                   'world_y': data['worldy'],
                   'los': data['los'],
                   'timestamp': data['timestamp']
                   }
        return csv_row

    def do(self):
        if not self.data_queue.empty():
            with open(self.raw_file, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields)
                while not self.data_queue.empty():
                    csv_row = self.convert_to_csv_row(self.data_queue.get())
                    if csv_row is not None:
                        w.writerow(csv_row)
                f.close()
    def enqueue(self, data):
        self.data_queue.put(data)


class IMULog:
    raw_file = settings.IMU_DATALOG_CONFIG['raw_file']
    csv_fields = ['IMU_ID', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'orientation_x', 'orientation_y', 'orientation_z', 'is_turing', 'timestamp']

    def __init__(self):
        self.data_queue = queue.Queue()
        self.init_csv()

    def init_csv(self):
        if not os.path.exists(self.raw_file):
            with open(self.raw_file, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields)
                w.writeheader()
                f.close()

    def convert_to_csv_row(self, data):
        if 'imuid' not in data:
            return None
        csv_row = {'IMU_ID': data['imuid'],
                   'acceleration_x': data['accx'],
                   'acceleration_y': data['accy'],
                   'acceleration_z': data['accz'],
                   'orientation_x': data['orix'],
                   'orientation_y': data['oriy'],
                   'orientation_z': data['oriz'],
                   'is_turing':data['isturing'],
                   'timestamp': data['timestamp']
                   }
        return csv_row

    def do(self):
        if not self.data_queue.empty():
            with open(self.raw_file, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields)
                while not self.data_queue.empty():
                    csv_row = self.convert_to_csv_row(self.data_queue.get())
                    if csv_row is not None:
                        w.writerow(csv_row)
                f.close()
    def enqueue(self, data):
        self.data_queue.put(data)

class DataLogService(Service):
    def __init__(self):
        super(DataLogService).__init__()
        self.tasks = [WiTracingLog(), CameraDetectionLog(), IMULog()]

    # core async function
    async def do(self):
        # log data every 1 seconds
        await asyncio.sleep(1)
        for task in self.tasks:
            task.do()

    def enqueue(self, data):
        for task in self.tasks:
            if isinstance(task, WiTracingLog) or isinstance(task, CameraDetectionLog) or isinstance(task, IMULog):
                task.enqueue(data)


if __name__ == "__main__":
    datalog = DataLogService()
    datalog.run()
