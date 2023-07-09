import numpy as np
import torch
import torch.nn as nn
import math


class RFCam_imu(nn.Module):
    def __init__(self, imu_feature_size,  num_classes):
        super(RFCam_imu, self).__init__()
        # Wifi encoder
        # tokenizer.vocab_size = 28996
        # embedding_dim = 16
        # hidden_size = 64
        self.embedding_layer = nn.Embedding(256, 8)

        # Feature extraction layers for IMU data
        self.imu_feature_extractor = nn.GRU(input_size=6,
                                            hidden_size=imu_feature_size,
                                            num_layers=2,
                                            dropout=0.2,
                                            batch_first=True)

        self.action_predictor = nn.Sequential(
            nn.Linear(imu_feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        wifi_name, wifi_rssi, imu_data, visual_data = obs

        # Process tracklets as a batch
        # Extract features from IMU data
        imu_features, _ = self.imu_feature_extractor(imu_data)
        batch_size, length, feature_size = imu_features.shape
        imu_features = imu_features.reshape(batch_size * length, feature_size)

        action = self.action_predictor(imu_features)
        action = action.reshape(batch_size, length)

        return action
