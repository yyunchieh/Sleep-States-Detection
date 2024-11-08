import sys, time
import torch

def trainer(num_epochs, 
            model, 
            loss,
            optimizer,
            train_loader,
            valid_loader,
            early_stopping,
            device):


    
    for epoch in range(num_epochs):
        
        model.train()
    
        epoch_time = 0
        loss_train_fn = 0
        n = 0
        
        for r, (x, y) in enumerate(train_loader):
    
            start = time.time()
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
    
            H, (h_T, c_T), y_hat = model(x)
            
            loss_train = loss(y_hat, y)
            loss_train.backward()
            
            optimizer.step()
    
            n += len(y)
            loss_train_fn += loss_train.item()*len(y) / n
            
            epoch_time += time.time()-start
         
            print("Epoch: {:>3d}; train iteration: {:>4d}; time: {:>6.2f} secs; loss: {:>6.4f}".format(epoch+1, r+1, epoch_time, loss_train_fn),  
                  end="\r", file=sys.stdout, flush=True)
 
    
    
        model.eval()
    
        epoch_time = 0
        loss_val_fn = 0
        n = 0
        
        for r, (x, y) in enumerate(valid_loader):
    
    
            start = time.time()
            
            x, y = x.to(device), y.to(device)
                
            with torch.no_grad(): 
                
                
                H, (h_T, c_T), y_hat = model(x)
                
                loss_val = loss(y_hat, y)
    
                n += len(y)
                loss_val_fn += loss_val.item()*len(y) / n
                
                epoch_time += time.time()-start
                
                print("Epoch: {:>3d}; valid iteration: {:>4d}; time: {:>6.2f} secs; loss: {:>6.4f}".format(epoch+1, r+1, epoch_time, loss_val_fn),  
                  end="\r", file=sys.stdout, flush=True)
        
        #Early Stopping
    
        
            
        if early_stopping(loss_val.item()): 
            
            print('\n Early stopping at epoch {:>3d}'.format(epoch +1))
        
            break


class EarlyStopping:
    
    def __init__(self, patience=10, verbose=False):
        
        self.patience = patience
        self.verbose = verbose
        self.counter=0
        self.best_loss = None

    def __call__(self, val_loss):
        
        if self.best_loss is None: 
            
            self.best_loss = val_loss
        
        elif val_loss < self.best_loss:
            
            self.best_loss = val_loss 
            self.counter = 0
        
        else:
            self.counter +=1
            
            if self.counter >= self.patience:
                
                if self.verbose: 
                    
                    print("\n Early stopping triggered.")

                return True
        
        return False
