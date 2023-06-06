import numpy as np
import torch
import torch.nn as nn
import math
'''

Fusion model v6. Use Conv1D instead of LSTM to extract features.

'''

class FusionModelv6(nn.Module):
    def __init__(self, wifi_feature_size, imu_feature_size, visual_feature_size, num_classes):
        super(FusionModelv6, self).__init__()
        self.embedding_layer = nn.Embedding(256, 8)
        self.imu_feature_extractor = nn.Conv1d(in_channels=6,
                                               out_channels=imu_feature_size,
                                               kernel_size=3,
                                               padding=1)
        self.wifi_encoder = nn.Conv1d(in_channels=2 * 8 * 20,
                                      out_channels=wifi_feature_size,
                                      kernel_size=3,
                                      padding=1)
        self.visual_feature_extractor = nn.Conv1d(in_channels=2,
                                                  out_channels=visual_feature_size,
                                                  kernel_size=3,
                                                  padding=1)

        self.matching_layer = nn.Conv1d(in_channels=imu_feature_size + wifi_feature_size + visual_feature_size,
                                        out_channels=1,
                                        kernel_size=3,
                                        padding=1)

        # self.pos_encoder = PositionalEncoding(imu_feature_size, 0.1)


        # Additional layer for the critic model
        self.critic_layer = nn.Linear(num_classes, 1)

    def forward(self, obs):
        wifi_name, wifi_rssi, imu_data, visual_data = obs

        wifi_name_embedding = self.embedding_layer(wifi_name)
        wifi_rssi_embedding = self.embedding_layer(wifi_rssi)
        wifi_input = torch.cat((wifi_name_embedding, wifi_rssi_embedding), dim=-1)
        batch_size, time_steps, _, _ = visual_data.shape
        wifi_input = wifi_input.view(batch_size, -1, time_steps)
        wifi_features = self.wifi_encoder(wifi_input)

        imu_data = imu_data.permute(0, 2, 1).contiguous()
        imu_features= self.imu_feature_extractor(imu_data)

        # Extract temporal features from each tracklet
        batch_size, time_steps, num_tracklets, visual_feature_size = visual_data.shape

        # Reshape visual_data to (batch_size * num_tracklets, time_steps, feature_size)
        visual_data = visual_data.permute(0, 2, 1, 3).contiguous().view(batch_size * num_tracklets, visual_feature_size,
                                                                        time_steps)

        # Process tracklets as a batch
        visual_features = self.visual_feature_extractor(visual_data)
        # visual_features = self.pos_encoder(visual_features)

        # Expand imu_features and wifi_features to match the shape of visual_features
        wifi_features_expanded = wifi_features.unsqueeze(2).expand(-1, -1, num_tracklets, -1).contiguous()
        batch_size, wifi_feature_size, num_tracklets, time_steps = wifi_features_expanded.shape
        wifi_features_reshaped = wifi_features_expanded.view(batch_size * num_tracklets, wifi_feature_size, time_steps).contiguous()
        # wifi_features_reshaped = self.pos_encoder(wifi_features_reshaped)

        imu_features_expanded = imu_features.unsqueeze(2).expand(-1, -1, num_tracklets, -1).contiguous()
        batch_size, imu_feature_size, num_tracklets, time_steps = imu_features_expanded.shape
        imu_features_reshaped = imu_features_expanded.view(batch_size * num_tracklets, imu_feature_size, time_steps).contiguous()

        # imu_features_reshaped = self.pos_encoder(imu_features_reshaped)

        # Concatenate imu_features_expanded and visual_features
        combined_features = torch.cat((visual_features, wifi_features_reshaped, imu_features_reshaped), dim=-2)

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
