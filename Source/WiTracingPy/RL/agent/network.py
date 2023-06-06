import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.
			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int
			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.
			Parameters:
				obs - observation to pass as input
			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output


    # def forward(self, obs):
    #     # if isinstance(obs, tuple):
    #     #     obs = obs[0]
    #     # Convert observation to tensor if it's a numpy array
    #     if isinstance(obs, np.ndarray):
    #         obs = torch.tensor(obs, dtype=torch.float)
    #
    #     if len(obs.shape) == 2:
    #         obs = torch.unsqueeze(obs, 0)
    #
    #         h_0 = Variable(torch.zeros(self.num_layers, obs.size(0), self.hidden_size))  # hidden state
    #         c_0 = Variable(torch.zeros(self.num_layers, obs.size(0), self.hidden_size))  # internal state
    #         # # Propagate input through LSTM
    #         output, (hn, cn) = self.lstm(obs, (h_0, c_0))  # lstm with input, hidden, and internal state
    #
    #         # hn = hn[0][0]
    #
    #         hn = hn.contiguous().view(-1)
    #
    #         activation1 = F.relu(self.layer1(hn))
    #         activation2 = F.relu(self.layer2(activation1))
    #         y = self.layer3(activation2)
    #         return y
    #
    #
    #     h_0 = Variable(torch.zeros(self.num_layers, obs.size(0), self.hidden_size))  # hidden state
    #     c_0 = Variable(torch.zeros(self.num_layers, obs.size(0), self.hidden_size))  # internal state
    #     # # Propagate input through LSTM
    #     output, (hn, cn) = self.lstm(obs, (h_0, c_0))  # lstm with input, hidden, and internal state
    #     # hn = hn[0][0]
    #     hn = hn.contiguous().view(obs.size(0), -1)
    #
    #     activation1 = F.relu(self.layer1(hn))
    #     activation2 = F.relu(self.layer2(activation1))
    #     y = self.layer3(activation2)
    #     return y

        # activation1 = F.relu(self.layer1(obs))
        # activation2 = F.relu(self.layer2(activation1))
        # y = self.layer3(activation2)
        # return y
