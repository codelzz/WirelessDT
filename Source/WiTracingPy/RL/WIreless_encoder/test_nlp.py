import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, num_words, embedding_dim, hidden_size):
        super(NeuralNetwork, self).__init__()

        # Define the embedding layer
        self.embedding_layer = nn.Embedding(num_words, embedding_dim)

        # Define the LSTM layer
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_size)

        # Define the dense output layer
        self.output_layer = nn.Linear(hidden_size, 2)

        self.result_layer = nn.Linear(12, 2)

    def forward(self, input_data):
        # Perform feedforward calculations
        embedding_output = self.embedding_layer(input_data)
        lstm_output, _ = self.lstm_layer(embedding_output)
        output = self.output_layer(lstm_output[-1])

        result = self.result_layer(output.view(-1)).unsqueeze(0)

        # Return output
        return result


# Define the dataset class
class TXNameDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the TXName string from the data
        tx_name = self.data[idx]

        # Convert the TXName string to a sequence of token ids
        tx_name_ids = torch.tensor(self.tokenizer.encode(tx_name, add_special_tokens=True))
        label = torch.tensor(idx, dtype=torch.float32)

        return tx_name_ids, label


# Define the tokenizer
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

# Define input data in the format of a list containing TXNames
input_data = ["TXName1", "TXName2", "TXName3"]

# Create the dataset and dataloader
dataset = TXNameDataset(input_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: (
    pad_sequence([item[0] for item in x], batch_first=True, padding_value=0),
    torch.tensor([item[1] for item in x], dtype=torch.float32)
))

# Create a neural network instance with 1000 possible words in the vocabulary, 16 embedding dimensions, and a hidden size of 64
model = NeuralNetwork(num_words=tokenizer.vocab_size, embedding_dim=16, hidden_size=64)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    running_loss = 0.0

    for batch in dataloader:
        TXName, label = batch
        # Zero the gradients
        optimizer.zero_grad()

        # Perform feedforward calculations and get the output
        output = model(TXName)

        # Define the target coordinates (for this example, we just use the batch index as the coordinates)
        target = torch.tensor([[label, label]], dtype=torch.float32)

        # Compute the loss
        loss = criterion(output, target)

        # Perform backpropagation and update the parameters
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

    # Print the average loss for the epoch
    print("Epoch %d, loss: %.3f" % (epoch + 1, running_loss / len(dataset)))

# Print the final output
with torch.no_grad():
    tx_name_ids = torch.tensor(tokenizer.encode("TXName1", add_special_tokens=True)).unsqueeze(0)
    output = model(tx_name_ids)
    print(output)

    tx_name_ids = torch.tensor(tokenizer.encode("TXName2", add_special_tokens=True)).unsqueeze(0)
    output = model(tx_name_ids)
    print(output)

    tx_name_ids = torch.tensor(tokenizer.encode("TXName3", add_special_tokens=True)).unsqueeze(0)
    output = model(tx_name_ids)
    print(output)
