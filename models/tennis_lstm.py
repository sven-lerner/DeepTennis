import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn

#very vanilla lstm

class TennisLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size,
                    num_layers=1, predict_mask=False, **kwargs):
        super(TennisLSTM, self).__init__()
        self.input_dim = input_dim + 14
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.predict_mask = predict_mask
        self.prematch_probs = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.1)
        if predict_mask:
            output_dim = 2
        else:
            output_dim = 1
        self.linear = nn.Linear(self.hidden_dim, output_dim)
    
    def set_prematch_probs(self, prematch_probs):
        self.prematch_probs = prematch_probs

    def get_blank_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))
    
    def forward(self, input):
        prematch_probs = self.prematch_probs.repeat(1,input.shape[1],1)
        input = torch.cat((input, prematch_probs), 2).transpose_(0, 1)
        hidden = self.get_blank_hidden()
        lstm_out, _ = self.lstm(input.view(len(input), self.batch_size, -1), hidden)
        linear_output = self.linear(lstm_out)
        out = torch.sigmoid(linear_output)
        if self.predict_mask:
            y_pred = out[:,:,0]
            mask = out[:,:,1]
            return y_pred.view(-1), mask.view(-1)
        else:
            return out.view(-1)


class TennisGRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size,
                    num_layers=1, predict_mask=False, **kwargs):
        super(TennisGRUNet, self).__init__()
        self.input_dim = input_dim + 14
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.predict_mask = predict_mask
        self.prematch_probs = None
        
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.1)
        if predict_mask:
            output_dim = 2
        else:
            output_dim = 1
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def set_prematch_probs(self, prematch_probs):
        self.prematch_probs = prematch_probs
        
    def get_blank_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, input):
        prematch_probs = self.prematch_probs.repeat(1,input.shape[1],1)
        input = torch.cat((input, prematch_probs), 2).transpose_(0, 1)
        hidden = self.get_blank_hidden()
        gru_out, _ = self.gru(input.view(len(input), self.batch_size, -1))
        linear_output = self.linear(gru_out)
        out = torch.sigmoid(linear_output)
        if self.predict_mask:
            y_pred = out[:,:,0]
            mask = out[:,:,1]
            return y_pred.view(-1), mask.view(-1)
        else:
            return out.view(-1)
