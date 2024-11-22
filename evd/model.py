import torch
from torch import nn


class LSTMModel(nn.Module):
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim,
                 num_layers,
                 num_classes):
        
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        
        H, (h_T, c_T) = self.lstm(x)
        y_hat = self.fc(h_T[-1,:,:].squeeze(0))
         
        return H, (h_T, c_T), y_hat 
