import torch
import torch.nn as nn
from custom_loss import CustomLoss
import matplotlib.pyplot as plt


class Train:
    def __init__(self, model, number_of_epochs: int, path_saving: str, device='cpu'):
        self.optimizer = None
        self.loss_f_1 = None
        self.loss_f = None
        self.losses = None
        self.path = path_saving
        self.model = model  # model to train
        self.number_of_epochs = number_of_epochs  # number of training epochs
        self.device = device  # device to train on

    def compile(self, loss_f="BCE_LL", optimizer='Adam',
                learning_rate=0.0001):
        # compile loss and optimizer for training
        self.loss_f = CustomLoss()
        if loss_f == "BCE_LL":
            self.loss_f_1 = nn.BCEWithLogitsLoss()
            # set BinaryCrossEntropy loss function for abnormal output as
            # operator's preference
        elif loss_f == "BCE":
            self.loss_f_1 = nn.BCELoss()
            # set BinaryCrossEntropy loss function as operator's preference for binary classification
        elif loss_f == "CE":
            self.loss_f_1 = nn.CrossEntropyLoss()
            # set CrossEntropy loss function as operator's preference for multiclass classification
        elif loss_f == "MSE":
            self.loss_f_1 = nn.MSELoss()
            # set Mean Squared Error loss function as operator's preference for regression problems
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=learning_rate)
            # set Adam optimizer as operator's preference

    def train(self, train_loader, verbose=True):
        # define the training loop
        self.losses = []
        for epoch in range(self.number_of_epochs):
            total_loss = 0.0  # at the beginning of each epoch, the loss will be assigned to zer0
            batch = 0
            for images, labels in train_loader:  # iterate through training examples
                batch += 1
                images = images.to(self.device).to(torch.float32)  # save the batch of images in device memory
                labels = labels.to(self.device).to(torch.float32)  # save the batch of labels in device memory

                predicts = self.model(images).squeeze_()
                # pass the image to the model to obtain the output then squeeze it
                loss_p = self.loss_f_1(predicts[:, :4], labels[:, :4])
                # compute loss between the training examples and the outputs of the model
                loss_a = self.loss_f(predicts[:, 4:], labels[:, 4:])
                loss = loss_a + loss_p
                # compute loss between the training examples and the outputs of the model
                loss.backward()
                # back propagate the loss
                self.optimizer.step()
                # compute the backpropagation step by optimizer function
                self.optimizer.zero_grad()
                total_loss += loss.item()
                # add loss of each batch to the total loss of the epoch
                if batch % 100 == 0:
                    print(f"Loss[batch = {batch}/{len(train_loader)}]= {loss.item()}"
                          f"  , total_loss = {total_loss}")
            self.losses.append(total_loss)
            self._save_(dynamic_save=True)
            if verbose:
                print(
                    f'For the epoch number {epoch + 1}: The training loss is {total_loss}.')
                # if the verbose flag is set to True, print the training loss value
        self._save_()
        return self.model  # return the trained model

    def n_parameters(self):  # compute the number of parameters
        total_par = 0
        for par in self.model.parameters():  # iterate through model parameters
            if par.requires_grad:  # compute only the trainable parameters
                total_par += par.numel()  # add the number of parameters of each layer to the total number of parameters
        print(f"Total number of trainable parameters for this model is {total_par}")

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(self.path + r"\loss_training.png")

    def _save_(self, dynamic_save=False):
        # save the model parameters
        if dynamic_save:
            torch.save(self.model.state_dict(), self.path + r"\model_dynamic.pt")
        else:
            torch.save(self.model.state_dict(), self.path + r"\model.pt")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        # load the preserved model parameters
        return self.model
