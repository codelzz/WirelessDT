
# datalog is a logger service saving data to local repo during simulation

import os
import csv
import asyncio
import queue
from service import Service
from settings import DATALOG_CONFIG

LOSS_FILENAME = DATALOG_CONFIG['loss_filename']
EVAL_FILENAME = DATALOG_CONFIG['eval_filename']
BASE_FILENAME = DATALOG_CONFIG['base_filename']


class TrainingLog:
    loss = {
        'tag':'loss',
        'csv_fields': ['iteration','x','y','power','loss'],
        'filename': LOSS_FILENAME,
        'queue': queue.Queue(),
    }
    
    eval = {
        'tag':'eval',
        'csv_fields': ['x','y','rss prediction', 'rss label'],
        'filename': EVAL_FILENAME,
        'queue': queue.Queue(),
    }
    
    base = {
        'tag':'base',
        'csv_fields': ['x','y','rss prediction', 'rss label'],
        'filename': BASE_FILENAME,
        'queue': queue.Queue(), 
    }

    def __init__(self):
        # self.loss_queue = queue.Queue()
        # self.eval_queue = queue.Queue()
        self.init_csv()

    # csv file operation
    def init_csv(self):
        for data in [self.loss, self.eval, self.base]:
            if not os.path.exists(data['filename']):
                with open(data['filename'], 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=data['csv_fields'])
                    w.writeheader()
                    f.close()

    def convert_to_csv_row(self, data, fieldnames):
        csv_row = {}
        for key in fieldnames:
            csv_row[key] = data[key]
        return csv_row

    def do(self):
        for data in [self.loss, self.eval, self.base]:
            if not data['queue'].empty():
                with open(data['filename'], 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=data['csv_fields'])
                    while not data['queue'].empty():
                        w.writerow(self.convert_to_csv_row(data['queue'].get(), fieldnames=data['csv_fields']))
                    f.close()

    def enqueue(self, data, tag):
        if tag == "loss":
            self.loss['queue'].put(data)
        elif tag == "eval":
            self.eval['queue'].put(data)
        elif tag == "base":
            self.base['queue'].put(data)

class DataLogService(Service):
    def __init__(self):
        super(DataLogService).__init__()
        self.tasks = [TrainingLog()]

    # core async function
    async def do(self):
    	# log data every 1 seconds
        await asyncio.sleep(1)
        for task in self.tasks:
            task.do()

    def enqueue(self, data, tag=None):
        for task in self.tasks:
            if isinstance(task, TrainingLog):
                task.enqueue(data, tag=tag)


if __name__ == "__main__":
    datalog = DataLogService()
    datalog.run()