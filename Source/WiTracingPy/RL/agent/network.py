import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.num_layers = 1
        self.hidden_size = 64
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)  # lstm
        self.layer1 = nn.Linear(self.hidden_size, 64)
        # self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        h_0 = Variable(torch.zeros(self.num_layers, obs.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, obs.size(0), self.hidden_size))  # internal state

        output, (hn, cn) = self.lstm(obs, (h_0, c_0))  # lstm with input, hidden, and internal state

        hn = hn.contiguous().view(-1, self.hidden_size)
        activation1 = F.relu(self.layer1(hn))
        activation2 = F.relu(self.layer2(activation1))
        y = self.layer3(activation2)
        return y


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
