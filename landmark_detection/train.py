import torch
import torch.nn as nn
from custom_loss import CustomLoss
import matplotlib.pyplot as plt


class Train:
    def __init__(self, model, number_of_epochs: int, path_saving: str, device='cpu'):
        self.val_losses_p = None
        self.val_losses_a = None
        self.val_losses = None
        self.optimizer = None
        self.loss_f_1 = None
        self.loss_f = None
        self.losses = None
        self.path = path_saving
        self.model = model  # model to train
        self.number_of_epochs = number_of_epochs  # number of training epochs
        self.device = device  # device to train on

    def initialize(self, loss_f="BCE_LL", optimizer='Adam',
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

    def train(self, train_loader, val_loader, verbose=True):
        # define the training loop
        self.losses, self.val_losses = [], []
        Length = len(train_loader)
        for epoch in range(self.number_of_epochs):
            total_loss, loss_angle, loss_position = 0.0, 0, 0  # at the beginning of each epoch, the loss will be assigned to zer0
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
                loss_angle += loss_a
                loss_position += loss_p
                # add loss of each batch to the total loss of the epoch
                if batch in [Length/4, Length/2, Length*3/4]:
                    print(f"Loss[batch = {batch}/{Length}]= {loss.item()}"
                          f"  , total_loss = {total_loss}")
                    print(f"Loss_angle = {loss_a} and Loss Points =  {loss_p}")
            self.losses.append(total_loss)
            self._save_(dynamic_save=True)
            if verbose:
                print(
                    f'For the epoch number {epoch + 1}: The training loss is {total_loss}.')
                print(f"Loss_angle = {loss_angle} and Loss Points =  {loss_position}")
                # if the verbose flag is set to True, print the training loss value
            with torch.no_grad():
                loss, loss_a, loss_p = 0, 0, 0
                total_loss = 0
                for images, labels in val_loader:
                    images = images.to(self.device).to(torch.float32)
                    labels = labels.to(self.device).to(torch.float32)
                    predicts = self.model(images).squeeze_()
                    # pass the image to the model to obtain the output then squeeze it
                    loss_p += self.loss_f_1(predicts[:, :4], labels[:, :4])
                    # compute loss between the training examples and the outputs of the model
                    loss_a += self.loss_f(predicts[:, 4:], labels[:, 4:])
                    loss = loss_a + loss_p
                    del images, labels
                self.val_losses.append(loss)
                self.val_losses_a.append(loss_a)
                self.val_losses_p.append(loss_p)
                print(
                    f'For the epoch number {epoch + 1}: The training loss is {total_loss}.')
                print(f"Loss_angle = {loss_a} and Loss Points =  {loss_p}")
        self._save_()
        return self.model  # return the trained model

    def n_parameters(self):  # compute the number of parameters
        total_par = 0
        for par in self.model.parameters():  # iterate through model parameters
            if par.requires_grad:  # compute only the trainable parameters
                total_par += par.numel()  # add the number of parameters of each layer to the total number of parameters
        print(f"Total number of trainable parameters for this model is {total_par}")

    def plot_loss(self, x, title: str, name_parameter: str):
        plt.plot(x)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(name_parameter)
        plt.savefig(self.path + rf"\{title}.png")

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
