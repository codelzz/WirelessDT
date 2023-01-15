import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import time
import pandas as pd
import numpy as np
import queue

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import settings
from udpthread.runnable import Runnable
from nn.deepar import create_model

# Historical csv file contains data that used for model training
# We can estimate the mean and variance of data distribution from historical data
HISTORICAL_CSV_FILE = settings.APP_CONFIG['historical_csv_file']
MODEL_CHECKPOINT_PATH = settings.APP_CONFIG['checkpoint_path']


class Predictor(Runnable):
    def __init__(self,
        on_predict,
        wait_time=0.001, 
        historical_csv_path=HISTORICAL_CSV_FILE, 
        checkpoint=MODEL_CHECKPOINT_PATH, 
        max_window_size=50,
        min_window_size=10):
        super(Predictor, self).__init__(wait_time=wait_time)
        self.mean, self.std, self.epsilon = None, None, None
        self.input_size, self.label_size = None, None
        self.columns = None
        self.historical_df = None
        self.load_historical_data(csv_path=historical_csv_path)
        # model
        self.model = None
        self.load_model(checkpoint=checkpoint)
        # data
        self.data_queue = queue.Queue()
        #
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.reach_min_window = False
        self.measurement = {}
        self.init_measurement()
        self.count = 0
        #
        self.on_predict = on_predict
        #
        self.print("Ready.")

    def summary(self):
        print('COLUMNS:', self.columns)
        print('MEAN:', list(self.mean))
        print('STD: ', list(self.std))
        print('LABEL SIZE: ', self.label_size)
        print('INPUT SIZE: ', self.input_size)
        print('RAW MEASUREMENT:', self.measurement)
        print('MAX WINDOW SIZE:', self.max_window_size)

    def load_historical_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.drop(columns=['t'])
        df = df.replace(-255.0, np.nan)
        df = df.interpolate('linear')
        self.epsilon = 1e-10
        self.mean = df.mean()
        self.std = df.std() + self.epsilon
        self.label_size = 3
        self.input_size = len(df.columns[self.label_size:])
        self.columns = df.columns
        self.historical_df = df
        
    def init_measurement(self):
        for column in self.columns[self.label_size:]:
            self.measurement[column] = []
        
    def add_measurement(self, tx, rssi):
        if tx in self.measurement.keys():
            self.measurement[tx].append(rssi)
            if len(self.measurement[tx]) > self.max_window_size:
                self.measurement[tx] = self.measurement[tx][1:]
                if len(self.measurement[tx]) > self.min_window_size:
                    self.reach_min_window = True

    def get_measurement(self):
        m = self.measurement.copy()
        min_size = self.max_window_size
        for k, v in m.items():
            min_size = min(len(v), min_size)
        for k, v in m.items():
            m[k] = v[-min_size:]
        return m
                
    def gen_input_seq(self):
        df = pd.DataFrame.from_dict(self.get_measurement())
        df = df.replace(-255.0, np.nan)
        df = df.interpolate('linear')
        # normalize
        df = (df - self.mean[self.label_size:]) / self.std[self.label_size:]
        return tf.convert_to_tensor([df.to_numpy()], dtype='float32')
    
    def load_model(self, checkpoint):
        self.model = create_model(dim_x=self.input_size, dim_z=self.label_size)
        # restore checkpoints if exist
        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint, max_to_keep=100)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restoring checkpoint from {}".format(ckpt_manager.latest_checkpoint))
            
    def predict(self):
        inputs = self.gen_input_seq()
        print(inputs.shape)
        mean, _ = self.model(inputs)
        return mean * self.std[:self.label_size] + self.mean[:self.label_size]

    def enqueue(self, data):
        self.data_queue.put(data)

    def do(self):
        if not self.data_queue.empty():
            data = self.data_queue.get()
            self.add_measurement(tx=data['txname'].lower(), rssi=data['rssi'])
            # prediction
            if self.ready_to_predict():
                p = self.predict()
                if p is None:
                    return
                self.on_predict(p[0,-1,:])

    def ready_to_predict(self):
        self.count+=1
        self.count = self.count % (self.input_size * 10)
        if self.count == 0:
            return True
        return False
