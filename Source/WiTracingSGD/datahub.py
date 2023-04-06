# datahub is a hub transmitting data between Unreal Project
# and other component via websocket.
# ref: https://websockets.readthedocs.io/en/stable/howto/patterns.html
import asyncio
import websockets
import json
import re
import settings
import numpy as np
import pandas as pd
from service import Service
from settings import DATAGEN_CONFIG, DATAHUB_CONFIG
from estimator import TXEstimator

# async def handle(websocket):
#     async for message in websocket:
#         print(message)
#         await websocket.send(message)

# Here we use simple consumer and producer pattern to send/recv message on same socket
# consumer: the function to handle received message
# producer: the function to handle sent message

HOST, PORT = DATAHUB_CONFIG['endpoint']

MAC = DATAGEN_CONFIG['mac']
RX_SAMPLE_SIZE = DATAGEN_CONFIG['rx_sample_size']
TX_SAMPLE_SIZE = DATAGEN_CONFIG['tx_sample_size']
TX_SAMPLE_SIGMA = DATAGEN_CONFIG['tx_sample_sigma']
SLEEP_TIME = DATAGEN_CONFIG['sleep_time']
EPOCHS = DATAGEN_CONFIG['epochs']

class Stage:
    BASELINE = 0
    TRAINING = 1
    EVALUATION = 2
    COMPLETED = 3


class DataHubService(Service):
    debug = False 

    def __init__(self, host=HOST, port=PORT, datagen=None, datalog=None):
        super(DataHubService).__init__()
        self.host = host
        self.port = port
        self.datagen = datagen
        self.estimator = None
        self.sync_lock = False
        self.init_estimator(datalog)
        self.stage = Stage.BASELINE

    def init_estimator(self, datalog):
        self.datagen.load_train_df(mac=MAC)
        self.estimator = TXEstimator(init_estimation = self.datagen.get_init_estimation(), datalog=datalog)
    
    # on message recv
    async def consumer(self, msg):
        data = json.loads(msg)
        self.print(f"[DBG] <<<")
        
        if self.stage == Stage.BASELINE:
            self.estimator.update_baseline(pred=data)
            if self.datagen.epochs > 1:
                self.stage = Stage.TRAINING
        
        elif self.stage == Stage.TRAINING:
            self.estimator.update_estimation(pred=data)
            if self.datagen.epochs > EPOCHS + Stage.TRAINING:
                self.stage = Stage.EVALUATION
        
        elif self.stage == Stage.EVALUATION:
            self.estimator.update_evaluation(pred=data)
            if self.datagen.epochs > EPOCHS + Stage.EVALUATION:
                self.stage = Stage.COMPLETED
        
        else:
            pass

         # if reach to the max epochs, switch to evaluation stage
        
        self.sync_lock = False # The loal
    

    # on message sent
    # There is two stage
    async def producer(self):
        if self.sync_lock:
            await asyncio.sleep(SLEEP_TIME) # How come this code is necessary for avoiding blocking
            return
        
        if self.stage == Stage.COMPLETED:
            return
    
        # load data for training
        if self.datagen.is_not_ready():
            self.datagen.load_train_df(mac=MAC)
        
        # generate sample
        rx_samples = self.datagen.get_samples(n=RX_SAMPLE_SIZE, drop=True, replace=False) 
        if rx_samples:
            self.estimator.update_rx_samples(samples=rx_samples) 

            data = {'rxi':rx_samples['index'],
                    'rxx':rx_samples['x'], 
                    'rxy':rx_samples['y'], 
                    'rxrssi': rx_samples['rssi'],
                    'txi':[],
                    'txx':[],
                    'txy':[],
                    'txpower':[],
                    'x': self.estimator.location[0],
                    'y': self.estimator.location[1],
                    'stage': self.stage}
            # In TRAINING Stage, the estimator generate gaussian distributed sample for training
            if self.stage == Stage.TRAINING:
                tx_samples = self.estimator.generate_samples(n=TX_SAMPLE_SIZE, sigmas=TX_SAMPLE_SIGMA)
                if tx_samples:
                    data['txi'] = tx_samples['index']
                    data['txx'] = tx_samples['x']
                    data['txy'] = tx_samples['y']
                    data['txpower'] = tx_samples['power']
            # In EVALUATION stage, no tx sample required, we use the final estimation as the result for evaluation
            elif self.stage == Stage.EVALUATION or self.stage == Stage.BASELINE:
                    data['txi'] = [0]
                    data['txx'] = [self.estimator.location[0]]
                    data['txy'] = [self.estimator.location[1]]
                    data['txpower'] = [self.estimator.power]
            msg = json.dumps(data)
            if self.debug:
                self.print(f"[DBG] >>> {rx_samples['index']}")
            return msg
        return None

    async def consumer_handler(self, websocket):
        async for msg in websocket:
            await self.consumer(msg)

    async def producer_handler(self, websocket):
        while True:
            msg = await self.producer()
            if msg:
                await websocket.send(msg)
                self.sync_lock = True

    async def handler(self, websocket):
        consumer_task = asyncio.create_task(self.consumer_handler(websocket))
        producer_task = asyncio.create_task(self.producer_handler(websocket))
        done, pending = await asyncio.wait(
            [consumer_task, producer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    async def core(self):
        self.print("[INF] starting...")
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # run forever

    def parse_message(self, msg):
        results = []
        try:
            msg_list = re.split('({[^}]*})', msg)[1::2]
            for m in msg_list:
                data = json.loads(m)
                data = {k.lower(): v for k, v in data.items()} # convert to lower
                if self.debug: 
                    self.print("[DBG] <<<\n" + str(data))
                results.append(data)
        except:
            self.print(f'[ERR] fail to parse:\n {msg}\n')
        return results

if __name__ == "__main__":
    HOST, PORT = settings.DATAHUB_CONFIG['endpoint']
    datahub = DataHubService(host=HOST, port=PORT)
    datahub.run()
