NETWORK_CONFIG = {
    "max_buffer_size": 65507,
    "server_endpoint": ("127.0.0.1", 8888),
    "client_endpoint": ("127.0.0.1", 8080),
}

APP_CONFIG = {
    "preprocessed_csv_file": "data/preprocessed_data.csv",
    "checkpoint_path": "checkpoints/deepAR",
    'training_history': "checkpoints/deepAR/history.csv",
    "training_epochs": 1,
    "data_size": 10000,
}

UNREAL_CONFIG = {
    'num_of_beacons': 22
}