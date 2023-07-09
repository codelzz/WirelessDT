import numpy as np
import torch
import torch.nn as nn
import math


class RFCam_visual(nn.Module):
    def __init__(self, visual_feature_size,  num_classes):
        super(RFCam_visual, self).__init__()
        # Wifi encoder
        # tokenizer.vocab_size = 28996
        # embedding_dim = 16
        # hidden_size = 64
        self.embedding_layer = nn.Embedding(256, 8)

        self.visual_feature_extractor = nn.GRU(input_size=2,
                                               hidden_size=visual_feature_size,
                                               num_layers=2,
                                               dropout=0.2,
                                               batch_first=True)

        self.dist_predictor = nn.Sequential(
            nn.Linear(visual_feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.angle_predictor = nn.Sequential(
            nn.Linear(visual_feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(visual_feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        wifi_name, wifi_rssi, imu_data, visual_data = obs
        if len(visual_data.shape) == 3:
            visual_data.unsqueeze(1)
        else:
            visual_data = visual_data[:, :, 0, :]

        # Process tracklets as a batch
        visual_features, _ = self.visual_feature_extractor(visual_data)
        batch_size, length, feature_size = visual_features.shape
        visual_features = visual_features.reshape(batch_size * length, feature_size)

        distance = self.dist_predictor(visual_features)
        angle = self.angle_predictor(visual_features)
        action = self.action_predictor(visual_features)

        distance = distance.reshape(batch_size, length)
        angle = angle.reshape(batch_size, length)
        action = action.reshape(batch_size, length)

        return distance, angle, action
