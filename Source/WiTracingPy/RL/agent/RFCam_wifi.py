import numpy as np
import torch
import torch.nn as nn
import math


class RFCam_wifi(nn.Module):
    def __init__(self, wifi_feature_size,  num_classes):
        super(RFCam_wifi, self).__init__()
        # Wifi encoder
        # tokenizer.vocab_size = 28996
        # embedding_dim = 16
        # hidden_size = 64
        self.embedding_layer = nn.Embedding(256, 8)

        self.wifi_encoder = nn.LSTM(input_size=2 * 8 * 20,
                                    hidden_size=wifi_feature_size,
                                    num_layers=1,
                                    batch_first=True)

        self.dist_predictor = nn.Sequential(
            nn.Linear(wifi_feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.angle_predictor = nn.Sequential(
            nn.Linear(wifi_feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(wifi_feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        wifi_name, wifi_rssi, imu_data, visual_data = obs
        # Wifi Encoder
        wifi_name_embedding = self.embedding_layer(wifi_name)
        wifi_rssi_embedding = self.embedding_layer(wifi_rssi)
        wifi_input = torch.cat((wifi_name_embedding, wifi_rssi_embedding), dim=-1)
        batch_size, time_steps, _, _ = visual_data.shape

        wifi_input = wifi_input.view(batch_size, time_steps, -1)
        wifi_features, _ = self.wifi_encoder(wifi_input)
        batch_size, length, feature_size = wifi_features.shape
        wifi_features = wifi_features.reshape(batch_size * length, feature_size)

        distance = self.dist_predictor(wifi_features)
        angle = self.angle_predictor(wifi_features)
        action = self.action_predictor(wifi_features)

        distance = distance.reshape(batch_size, length)
        angle = angle.reshape(batch_size, length)
        action = action.reshape(batch_size, length)

        return distance, angle, action
