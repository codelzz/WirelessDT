import torch
import torch.nn as nn


class FusionModelv2(nn.Module):
    def __init__(self, imu_feature_size, visual_feature_size, lstm_hidden_size, max_pedestrian_detections, num_classes):
        super(FusionModelv2, self).__init__()

        # Feature extraction layers for IMU data
        self.imu_feature_extractor = nn.LSTM(input_size=6,
                                             hidden_size=imu_feature_size,
                                             num_layers=1,
                                             batch_first=True)

        # LSTM layer for extracting temporal features from each tracklet
        self.visual_feature_extractor = nn.LSTM(input_size=2,
                                                hidden_size=visual_feature_size,
                                                num_layers=1,
                                                batch_first=True)

        # LSTM layer for capturing temporal dependencies
        self.lstm = nn.LSTM(input_size=imu_feature_size + max_pedestrian_detections * visual_feature_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=1,
                            batch_first=True)

        # Fully connected layer to learn the similarity metric
        self.matching_layer = nn.Sequential(
            nn.Linear(imu_feature_size + visual_feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Additional layer for the critic model
        self.critic_layer = nn.Linear(num_classes, 1)

        # Output layer for generating matching probabilities
        # self.output_layer = nn.Linear(lstm_hidden_size, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, obs):
        imu_data, visual_data = obs
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

        # Expand imu_features to match the shape of visual_features
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




        # Concatenate IMU and visual features
        # combined_features = torch.cat((imu_features, visual_features), dim=-1)

        # LSTM layer
        # lstm_output, _ = self.lstm(combined_features.permute(1,0,2))
        #
        # # Output layer
        # output = self.output_layer(lstm_output)
        #
        # matching_probs = self.sigmoid(output)

        # return matching_probs
