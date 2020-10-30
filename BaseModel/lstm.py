#!/user/bin/env python
# coding=utf-8
"""
@file: lstm.py
@author: zwt
@time: 2020/10/30 15:26
@desc: https://github.com/nikhilbarhate99/Char-RNN-PyTorch/blob/master/CharRNN.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, output_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())



def train():
    pass






