import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModel(nn.Module):
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim,
                 num_layers,
                 num_classes):
        
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, lengths):
        
        packed_input = pack_padded_sequence(x, lengths, batch_first = True, enforce_sorted=False)

        packed_output, (h_T, c_T) = self.lstm(packed_input)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)
    
        y_hat = self.fc(h_T[-1,:,:])
         
        return output, (h_T, c_T), y_hat 

seq_lengths = [20, 15, 10]

max_len = max(seq_lengths)
batch_size = len(seq_lengths)
input_dim = 10

test_input = torch.randn(batch_size, max_len, input_dim)
lengths = torch.tensor(seq_lengths)
model = LSTMModel(input_dim, hidden_dim=50, num_layers=2, num_classes=5)

output, (h_T, c_T), y_hat = model(test_input, lengths)

print("Logits (y_hat)", y_hat)