DATAHUB_CONFIG = {
    "endpoint": ("localhost",8000)
}

DATAGEN_CONFIG = {
    'data_dir': './dataset/',
    'train_dir':'./dataset/train/',
    'eval_dir':'./dataset/eval/',
    'mac':'20-20-00-00-00-06',
    'rx_sample_size': 30,
    'tx_sample_size': 10,
    'tx_sample_sigma': (200,200,3),
    'sleep_time': 0.1,
    'epochs':25,
    'rss_window': 100,
}

DATALOG_CONFIG = {
    'loss_filename': './log/loss/' + DATAGEN_CONFIG['mac'].replace(':','-') + '.csv',
    'eval_filename': './log/eval/' + DATAGEN_CONFIG['mac'].replace(':','-') + '.csv',
    'base_filename': './log/base/' + DATAGEN_CONFIG['mac'].replace(':','-') + '.csv'
}

TXESTIMATOR_CONFIG = {
    'lr_x':1, # learning rate for x
    'lr_y':1, # learning rate for y
    'lr_p':1, # learning rate for power
    'min_rssi': -95,
    'nlos_loss_coef': 10,
}