import numpy as np
import torch
import torch.nn as nn
import math

'''

Fusion model v3. Input with 3 LSTM feature extractor and FC layer as the fusing layer

'''

class FusionModelv9_No_WIFI(nn.Module):
    def __init__(self, wifi_feature_size, imu_feature_size, visual_feature_size, num_classes):
        super(FusionModelv9_No_WIFI, self).__init__()
        # Wifi encoder
        # tokenizer.vocab_size = 28996
        # embedding_dim = 16
        # hidden_size = 64
        self.embedding_layer = nn.Embedding(256, 8)
        # self.rssi_embedding_layer = nn.Embedding(256, embedding_dim)
        # Define the LSTM layer
        # wifi_feature_size = 64
        # self.wifi_encoder = nn.GRU(input_size=2 * 8 * 20,
        #                                      hidden_size=wifi_feature_size,
        #                                      num_layers=1,
        #                                      batch_first=True)


        # Feature extraction layers for IMU data
        self.imu_feature_extractor = nn.GRU(input_size=6,
                                             hidden_size=imu_feature_size,
                                             num_layers=2,
                                            dropout=0.2,
                                             batch_first=True)

        # LSTM layer for extracting temporal features from each tracklet
        self.visual_feature_extractor = nn.GRU(input_size=2,
                                                hidden_size=visual_feature_size,
                                                num_layers=2,
                                               dropout=0.2,
                                                batch_first=True)

        # Fully connected layer to learn the similarity metric
        self.matching_layer = nn.Sequential(
            nn.Linear(imu_feature_size + visual_feature_size, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Additional layer for the critic model
        self.critic_layer = nn.Linear(num_classes, 1)

        # Output layer for generating matching probabilities
        # self.output_layer = nn.Linear(lstm_hidden_size, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, obs):
        wifi_name, wifi_rssi, imu_data, visual_data = obs
        # Wifi Encoder
        # wifi_name_embedding = self.embedding_layer(wifi_name)
        # wifi_rssi_embedding = self.embedding_layer(wifi_rssi)
        # wifi_input = torch.cat((wifi_name_embedding, wifi_rssi_embedding), dim=-1)
        batch_size, time_steps, _, _ = visual_data.shape
        # wifi_input = wifi_input.view(batch_size, time_steps, -1)
        # wifi_features, _ = self.wifi_encoder(wifi_input)

        # Extract features from IMU data
        imu_features, _ = self.imu_feature_extractor(imu_data)

        # Extract temporal features from each tracklet
        batch_size, time_steps, num_tracklets, feature_size = visual_data.shape

        # Reshape visual_data to (batch_size * num_tracklets, time_steps, feature_size)
        visual_data = visual_data.permute(0, 2, 1, 3).contiguous().view(batch_size * num_tracklets, time_steps, feature_size)

        # Process tracklets as a batch
        visual_features, _ = self.visual_feature_extractor(visual_data)

        # Reshape visual_features back to (batch_size, time_steps, num_tracklets * visual_feature_size)
        # visual_features = visual_features.view(batch_size, num_tracklets, time_steps, -1).permute(0, 2, 1, 3).contiguous().view(batch_size, time_steps, -1)
        visual_features = visual_features.view(batch_size, num_tracklets, time_steps, -1).permute(0, 2, 1,
                                                                                                  3).contiguous()

        # Expand imu_features and wifi_features to match the shape of visual_features
        # wifi_features = wifi_features.unsqueeze(2).expand(-1, -1, num_tracklets, -1)
        imu_features_expanded = imu_features.unsqueeze(2).expand(-1, -1, num_tracklets, -1)

        # Concatenate imu_features_expanded and visual_features
        combined_features = torch.cat((imu_features_expanded, visual_features), dim=-1)

        # Reshape combined_features to (-1, imu_feature_size + visual_feature_size)
        combined_features = combined_features.view(-1, imu_features.shape[-1] + visual_features.shape[-1])

        # Compute similarity scores
        similarity_scores = self.matching_layer(combined_features)

        # Reshape similarity_scores back to (batch_size, time_steps, num_tracklets)
        similarity_scores = similarity_scores.view(batch_size, time_steps, num_tracklets)

        # Compute matching probabilities using softmax
        matching_probs = torch.softmax(similarity_scores, dim=-1)

        # Flatten the matching_probs tensor
        flattened_matching_probs = matching_probs.view(batch_size * time_steps, -1)

        # Apply the critic layer
        flattened_critic_output = self.critic_layer(flattened_matching_probs)

        # Reshape the critic_output back to the required shape
        critic_output = flattened_critic_output.view(batch_size, time_steps, 1)

        return matching_probs, critic_output