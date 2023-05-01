import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, imu_feature_size, visual_feature_size, lstm_hidden_size, max_pedestrian_detections, num_classes):
        super(FusionModel, self).__init__()

        # Feature extraction layers for IMU and visual data
        self.imu_feature_extractor = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, imu_feature_size)
        )

        self.visual_feature_extractor = nn.Sequential(
            nn.Linear(2 * max_pedestrian_detections, 128),
            nn.ReLU(),
            nn.Linear(128, visual_feature_size)
        )

        # LSTM layer for capturing temporal dependencies
        self.lstm = nn.LSTM(input_size=imu_feature_size + visual_feature_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=2,
                            batch_first=True)

        # Output layer for generating matching probabilities
        self.output_layer = nn.Linear(lstm_hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, obs):
        imu_data, visual_data = obs
        # Extract features from IMU and visual data
        imu_features = self.imu_feature_extractor(imu_data)
        visual_features = self.visual_feature_extractor(visual_data)

        # Concatenate IMU and visual features
        combined_features = torch.cat((imu_features, visual_features), dim=-1)

        # LSTM layer
        lstm_output, _ = self.lstm(combined_features.permute(1,0,2))

        # Output layer
        output = self.output_layer(lstm_output)

        matching_probs = self.sigmoid(output)

        return matching_probs
