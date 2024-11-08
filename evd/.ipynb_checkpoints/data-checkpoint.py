from torch.utils.data import Dataset

class TimeseriesDataset(Dataset): 
    
    def __init__(self, 
                 X, 
                 y, 
                 seq_len=1,
                 transform=None):
        
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, idx):

        data_X = self.X[idx:idx+self.seq_len]
        data_y = self.y[idx+self.seq_len-1]
        
        if self.transform:
            
            data_X, data_y = self.transform(data_X, data_y)
        
        return (data_X, data_y)
