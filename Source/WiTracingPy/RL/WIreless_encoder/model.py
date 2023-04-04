import torch
from torch import nn


class WirelessEncoder(nn.Module):
    def __init__(self, num_words, embedding_dim, hidden_size, max_length=1024):
        super(WirelessEncoder, self).__init__()
        # Define the embedding layer
        self.embedding_layer = nn.Embedding(num_words, embedding_dim)
        self.rssi_embedding_layer = nn.Embedding(256, embedding_dim)

        # Define the LSTM layer
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.rssi_layer = nn.Linear(20*embedding_dim, 512)

        # Define the dense output layer
        self.output_layer = nn.Linear(hidden_size, 2)

        self.result_layer1 = nn.Linear(max_length*2, 512)
        self.result_layer2 = nn.Linear(1024, 128)
        self.result_layer3 = nn.Linear(128, 3)

    def forward(self, input_data):
        txname, rssis = input_data
        # Perform feedforward calculations
        embedding_output = self.embedding_layer(txname)
        rssi_embedding_output = self.rssi_embedding_layer(rssis)

        lstm_output, _ = self.lstm_layer(embedding_output)
        rssi_output = self.rssi_layer(rssi_embedding_output.flatten(start_dim=1))

        output = self.output_layer(lstm_output)

        x = self.result_layer1(output.flatten(start_dim=1))
        x = torch.cat((x, rssi_output), 1)

        x = self.result_layer2(x)
        result = self.result_layer3(x)

        # Return output
        return result
