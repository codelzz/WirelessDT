import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)