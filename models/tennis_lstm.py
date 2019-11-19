import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn

#very vanilla lstm

class TennisLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(TennisLSTM, self).__init__()
        self.input_dim = input_dim + 14
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.prematch_probs = None
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        self.linear = nn.Linear(self.hidden_dim, output_dim)
        
    def init_hidden(self, prematch_probs):
        self.prematch_probs = prematch_probs.unsqueeze(0)
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        prematch_probs = self.prematch_probs.repeat(1,input.shape[1],1)
        input = torch.cat((input, prematch_probs), 2)
        # lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        lstm_out, self.hidden = self.lstm(input)
        
        # assert np.sum(torch.isnan(lstm_out).detach().numpy()) < 1, "hit a nan"
        linear_output = self.linear(lstm_out)
        # assert np.sum(torch.isnan(linear_output).detach().numpy()) < 1, "hit a nan"
        y_pred = torch.sigmoid(linear_output)
        # assert np.sum(torch.isnan(linear_output).detach().numpy()) < 1, "hit a nan"
        return y_pred.view(-1)
