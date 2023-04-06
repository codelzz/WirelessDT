import numpy as np
import pandas as pd
import glob
import signals
from settings import TXESTIMATOR_CONFIG, DATAGEN_CONFIG

LEARNING_RATE_X = TXESTIMATOR_CONFIG['lr_x']
LEARNING_RATE_Y = TXESTIMATOR_CONFIG['lr_y']
LEARNING_RATE_P = TXESTIMATOR_CONFIG['lr_p']
MIN_RSSI        = TXESTIMATOR_CONFIG['min_rssi']
NLOS_LOSS_COEF  = TXESTIMATOR_CONFIG['nlos_loss_coef']
DATA_DIR        = DATAGEN_CONFIG['train_dir']
EVAL_DIR        = DATAGEN_CONFIG['eval_dir']
MAC             = DATAGEN_CONFIG['mac']

class TXEstimator:
    def __init__(self, init_estimation, datalog=None):
        self.datalog = datalog
        self.location=[init_estimation['x'], init_estimation['y'], 100]
        self.power=init_estimation['power']
        assert len(self.location) == 3, "Invalid Dimension of Location"
        print(f"{'[TXEstimator]':18} [INF] TXEstimator -> location={self.location} | power={self.power}")

        self.txi_samples = None
        self.txx_samples = None
        self.txy_samples = None
        self.txp_samples = None
        
        self.rxi_samples = None
        self.rxx_samples = None
        self.rxy_samples = None
        self.rxr_samples = None
        
        self.iteration = 0
        self.loss_val = 0
        
        self.learn_rate = {'x': LEARNING_RATE_X, 'y': LEARNING_RATE_Y, 'power': LEARNING_RATE_P}
        self.sample_weights = None
        
    def generate_samples(self, n=10, sigmas=(1,1,1)):
        x_sigma, y_sigma, p_sigma = sigmas
        x, y, p = self.location[0], self.location[1], self.power
        self.txi_samples = np.array([i for i in range(0,n)])
        self.txx_samples = np.random.normal(x, x_sigma, n)
        self.txy_samples = np.random.normal(y, y_sigma, n)
        self.txp_samples = np.random.normal(p, p_sigma, n)
        return {'index': self.txi_samples.tolist(),'x':list(self.txx_samples),'y':list(self.txy_samples),'power':list(self.txp_samples), 'size':n}
    
    def update_rx_samples(self, samples):
        self.rxi_samples = np.array(samples['index'])
        self.rxx_samples = np.array(samples['x'])
        self.rxy_samples = np.array(samples['y'])
        self.rxr_samples = np.array(samples['rssi'])
        
    def update_estimation(self, pred):
        info = {'txi': int, 'rxi': int, 'rssi': int} 
        df = pd.DataFrame(columns=info.keys()).astype(info)
        df['txi'] = np.array(pred['txi'])
        df['rxi'] = np.array(pred['rxi'])
        df['rssi'] = np.array(pred['rxrssi'])
 
        losses = {}
        tx_indices = df['txi'].unique()
        for txi in tx_indices:
            sub_df = df[df['txi'] == txi]
            rxi, rssi = np.array(sub_df['rxi']), np.array(sub_df['rssi'])
            sample_matched = np.alltrue(rxi == self.rxi_samples)
            if sample_matched:
                losses[txi] = self.loss(pred = rssi, label = self.rxr_samples)
                
        if bool(losses):
            # Get the most matching param
            min_txi = min(losses, key=lambda k: losses[k])
            self.learn(pred={'x':self.txx_samples[min_txi], 
                             'y':self.txy_samples[min_txi], 
                             'power':self.txp_samples[min_txi]})
            self.loss_val = min(losses.values())
            print(self.location, self.power, min(losses.values()))
            
            # log result
            if self.datalog:
                self.datalog.enqueue(self.get_estimation(), tag='loss')
            
    def update_evaluation(self, pred):
        if self.datalog:
            for row in list(zip(self.rxx_samples, self.rxy_samples, self.rxr_samples, pred['rxrssi'])):
                x,y,l,p = row
                self.datalog.enqueue({'x':x,'y':y,'rss label':l, 'rss prediction': p}, tag='eval')
                
    def update_baseline(self, pred):
        if self.datalog:
            for row in list(zip(self.rxx_samples, self.rxy_samples, self.rxr_samples, pred['rxrssi'])):
                x,y,l,p = row
                self.datalog.enqueue({'x':x,'y':y,'rss label':l, 'rss prediction': p}, tag='base')
        
    def get_estimation(self):
        return {'iteration': self.iteration, 
                'x': self.location[0], 
                'y': self.location[1], 
                'power': self.power,
                'loss': self.loss_val}

    def loss(self, pred, label):
        # Non Line of Sight Loss
        # nlos_loss = np.sum(pred < MIN_RSSI) * NLOS_LOSS_COEF
        # Line of Sight Loss
        types = {'pred':int, 'label': int}
        df = pd.DataFrame(columns=types.keys()).astype(types)
        df['pred'], df['label'] = pred, label
        # convert signal to linear space (dBm -> mW)
        sample_weights = self.get_sample_weights_by_labels(labels=label)
        distance_weights = self.get_distance_weights_by_labels(labels=label)

        # df['pred']  = df['pred'].map(signals.dBmTomW)
        # df['label'] = df['label'].map(signals.dBmTomW)
        return np.sum(np.power(distance_weights, 4) * sample_weights * np.power(df['pred'] - df['label'], 2))

        #filtered_df = df[df['pred'] >= MIN_RSSI]
        #los_loss = np.sum(np.power(filtered_df['pred'] - filtered_df['label'], 2))
        #return los_loss + nlos_loss
    
    def learn(self, pred):
        x,  y,  p = pred['x'], pred['y'], pred['power']
        x_, y_, p_= self.location[0],self.location[1],self.power
        
        lr = self.get_learing_rate()

        self.location[0] += lr['x']     * (x - x_)
        self.location[1] += lr['y']     * (y - y_)
        self.power       += lr['power'] * (p - p_)
            
        self.iteration += 1
    
    def get_learing_rate(self):
        gamma = 0.99  
        lr = {}
        for key, value in self.learn_rate.items():
            lr[key] = value * np.power(gamma, self.iteration)
        return lr
    
    def get_sample_weights(self):
        # get sample weight according to dataset
        # sample weight is determinted by the RSS likelihood from training set
        if self.sample_weights is None:
            # load dataset
            def load():
                csvfiles = glob.glob(DATA_DIR + "/*.csv")
                csvfiles = [csvfile[-21:] for csvfile in csvfiles]
                dfs = []
                for csvfile in csvfiles:
                    path = f"{DATA_DIR}/{csvfile}"
                    dfs.append(pd.read_csv(path))
                return pd.concat(dfs)
            df = load()
            # calculate sample weight
            pct = df.RSS.value_counts() / len(df)
            # apply likelihood filter, remove 1% outliner
            lf = 1 # lf=1 not remove any outliner
            for i in range(0, len(pct)):
                total = np.sum(pct[:i])
                if total > lf:
                    pct = pct[:i]
                    break
            # compute weight
            self.sample_weights = 1.0 / pct
        return self.sample_weights
    
    def get_distance_weights_by_labels(self, labels):
        weights = 1.0 / signals.calc_distance_with(self.power, labels)
        return weights
    
    def get_sample_weights_by_labels(self, labels):
        sample_weights = self.get_sample_weights()
        weights = []
        for RSS in labels:
            if RSS in sample_weights.index:
                weights.append(sample_weights[RSS])
            else:
                weights.append(0)
        return weights
             
    
    def reset(self):
        self.txi_samples = None
        self.txx_samples = None
        self.txy_samples = None
        self.txp_samples = None
        self.rxi_samples = None
        self.rxx_samples = None
        self.rxy_samples = None
        self.rxr_samples = None
        self.iteration = 0
        self.loss = 0
