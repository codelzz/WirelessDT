import numpy as np
import pandas as pd
import glob
import signals
from scipy.optimize import least_squares
from settings import DATAGEN_CONFIG

DATA_DIR = DATAGEN_CONFIG['train_dir']
RSS_WINDOW = DATAGEN_CONFIG['rss_window']

class DataGenerator:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.train_df = None
        self.mac = None
        self.samples = None
        self.epochs = 0
        
    def load_train_df(self, mac):
        filename = mac.replace(':','-') + '.csv'
        self.mac = mac
        self.train_df = pd.read_csv(self.data_dir + filename)
        
    def pick_samples(self, n, drop=True, replace=True):        
        if self.train_df is not None:
            if len(self.train_df) < n:
                self.train_df = None
                self.samples = None
                self.epochs += 1
                return False
            self.samples = self.train_df.sample(n=n, replace=replace)
            if drop:
                self.train_df = self.train_df.drop(list(set(self.samples.index)))
            return True
        return False
    
    def get_samples(self, n, drop=False, replace=True):
        self.pick_samples(n=n, drop=drop, replace=replace)
        if self.samples is not None:
            i = list(self.samples.index)
            x = list(self.samples.UE_X)
            y = list(self.samples.UE_Y)
            rssi = list(self.samples.RSS)
            return {"index":i, "x":x, "y": y, "rssi":rssi, "size": n}
        return None
    
    def is_not_ready(self):
        return self.train_df is None
    
    def get_init_estimation(self, rss_window=RSS_WINDOW):
        if self.train_df is None:
            return 
                
        data = self.train_df[self.train_df.RSS > self.train_df.RSS.max() - rss_window][['UE_X','UE_Y','RSS']].to_numpy()
        data[:,:2] *= 0.01 # convert x y from UE unit to meter unique
        
        def cost(state, observation):
            x0, y0, pt_mW = state
            x1, y1, rss_dBm = observation
            d = signals.calc_distance_with(pt_mW=pt_mW, rss_dBm=rss_dBm)
            return np.power(np.sqrt((x0-x1)**2 + (y0-y1)**2) - d, 2)
        
        def fit(x):
            c = 0
            for observation in data:
                c += cost(x, observation)
            return c
        
        init_state = (data[:,0].mean(), data[:,1].mean(), 0.01)
        res = least_squares(fit, init_state)
        x = res.x[0] * 100 # to UE unique
        y = res.x[1] * 100 # to UE unique
        p = res.x[2] + 45 # Add System Gain
        return {'x':x, 'y':y, 'power': p}
    
if __name__ == "__main__":
    fdg = DataGenerator()
    fdg.load_train_df(mac='20:20:00:00:00:03')
    print(fdg.train_df[:1])