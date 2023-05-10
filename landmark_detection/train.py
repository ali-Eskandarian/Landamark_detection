import torch
import torch.nn as nn

class Train():
  
    def __init__(self, model, number_of_epochs : int, device):

        self.model = model
        self.number_of_epochs = number_of_epochs
        self.device = device

    def compile(self, loss_f="BCE_LL", optimizer='Adam', learning_rate=0.0001):
       
        if loss_f == "BCE_LL":
            self.loss_f = nn.BCEWithLogitsLoss()             
        elif  loss_f=="BCE":
           self.loss_f = nn.BCELoss()
        elif loss_f=="CE":
            self.loss_f = nn.CrossEntropyLoss()
        elif loss_f=="MSE":
            self.loss_f = nn.MSELoss()
        
        if optimizer == "Adam":
           self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train(self, train_loader, verbose=True):
        
        for epoch in range(self.number_of_epochs):
            total_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(images)
                preds = preds.squeeze_()
                loss = self.loss_f(preds, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
    
                total_loss += loss.item()
  
            """self.model.eval()
            val_total_loss = 0
            for val_images, val_labels in val_loader:
                val_images = val_images.to(self.device)
                val_labels = val_labels.to(self.device)

                val_preds = self.model(val_images)
                val_preds = val_preds.squeeze_()
                val_loss = self.loss_f(val_preds, val_labels)
    
                val_total_loss += val_loss.item()"""

            if verbose==True:
                print(f'For the epoch number {epoch+1}: The training loss is {total_loss}.')
        
        return self.model
    
    def n_parameters(model):
        
        total_par = 0
        for par in model.parameters():
            if par.requires_grad :
                total_par += par.numel()

        print(f"Total number of trainable paramaters for this model is {total_par}")

    def save_model(self, path : str):

        torch.save(self.model.state_dict(), path)

    def load_model(model, path):
        
        model.load_state_dict(torch.load(path))

        return model