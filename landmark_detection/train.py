import torch
import torch.nn as nn
from custom_loss import CustomLoss

class Train():                                                  
  
    def __init__(self, model, number_of_epochs : int, device = 'cpu'):

        self.model = model                                                                      # model to train
        self.number_of_epochs = number_of_epochs                                                # number of training epochs
        self.device = device                                                                    # device to train on

    def compile(self, loss_f="BCE_LL", optimizer='Adam', learning_rate=0.0001):                 # compile loss and optimizer for training
       
        self.loss_f = CustomLoss()
        if loss_f == "BCE_LL":
            self.loss_f_1 = nn.BCEWithLogitsLoss()                                                # set BinaryCrossEntropy loss function for non normal output as operator's preference
        
        elif  loss_f=="BCE":
           self.loss_f_1 = nn.BCELoss()                                                           # set BinaryCrossEntropy loss function as operator's preference for binary classification
        
        elif loss_f=="CE":
            self.loss_f_1 = nn.CrossEntropyLoss()                                                 # set CrossEntropy loss function as operator's preference for multiclass classification
        
        elif loss_f=="MSE":
            self.loss_f_1 = nn.MSELoss()                                                          # set Mean Squared Error loss function as operator's preference for regression problems
        
        if optimizer == "Adam":
           self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)         # set Adam optimizer as operator's preference
    
    def train(self, train_loader, verbose=True):                                                # define the training loop
        
        for epoch in range(self.number_of_epochs):
            total_loss = 0.0                                                                    # at the beginning of each epoch, the loss will be assigned to zer0
            batch = 0
            for images, labels in train_loader:                                                  # iterate through training examples
                batch += 1
                images = images.to(self.device).to(torch.float32)                                               # save the batch of images in device memory
                labels = labels.to(self.device)  .to(torch.float32)                                                # save the batch of labels in device memory

                preds = self.model(images)                                                      # pass the image to the model to obtain the output
                preds = preds.squeeze_()                                                        # squeeze the outputs of the model to be more convenient to use
                loss_p = self.loss_f_1(preds[:, :4], labels[:, :4])                             # compute loss between the training examples and the outputs of the model
                loss_a = self.loss_f(preds[:, 4:], labels[:, 4:])
                loss = loss_a + loss_p                               # compute loss between the training examples and the outputs of the model
                loss.backward()                                                                 # back propagate the loss
                self.optimizer.step()                                                           # compute the backpropagation step by optimizer function
                self.optimizer.zero_grad()
    
                total_loss += loss.item()                                                       # add loss of each batch to the total loss of the epoch
                if batch%1000 == 0:
                    print(f"Loss[batch = {batch}]= {loss.item()}  , total_loss = {total_loss}")
            if verbose==True:                                                                   
                print(f'For the epoch number {epoch+1}: The training loss is {total_loss}.')    # if the verbose flag is set to True, print the training loss value
        
        return self.model                                                                       # return the trained model
    
    def n_parameters(model):                                                                    # compute the number of parameters
        
        total_par = 0
        for par in model.parameters():                                                          # iterate through model parameters
            if par.requires_grad :                                                              # compute only the trainable parameters
                total_par += par.numel()                                                        # add the number of parameters of each layer to the total number of parameters

        print(f"Total number of trainable paramaters for this model is {total_par}")

    def save_model(self, path : str):

        torch.save(self.model.state_dict(), path)                                               # save the model parameters

    def load_model(model, path):
        
        model.load_state_dict(torch.load(path))                                                 # load the presaved model parameters

        return model