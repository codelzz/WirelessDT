from torch import nn


class WirelessEncoder(nn.Module):
    def __init__(self, num_words, embedding_dim, hidden_size, max_length=1024):
        super(WirelessEncoder, self).__init__()
        # Define the embedding layer
        self.embedding_layer = nn.Embedding(num_words, embedding_dim)

        # Define the LSTM layer
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

        # Define the dense output layer
        self.output_layer = nn.Linear(hidden_size, 2)

        self.result_layer = nn.Linear(max_length*2, 3)

    def forward(self, input_data):
        # Perform feedforward calculations
        embedding_output = self.embedding_layer(input_data)
        lstm_output, _ = self.lstm_layer(embedding_output)

        output = self.output_layer(lstm_output)
        # print(output.shape)

        result = self.result_layer(output.flatten(start_dim=1))

        # Return output
        return result
