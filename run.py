from load_data import Load_data
import os
import torch
import matplotlib.pyplot as plt
from model import Model
from train import Train

class Run():
    def __init__(self, device, batch_size, number_of_epochs):
        self.device = device
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

    def dataset(self):
        main_folder_path = os.getcwd()
        load = Load_data(self.device, main_folder_path, 'train_data', 'test_data')
        train_loader, val_loader, test_loader = load.load(self.batch_size, 0.8)

        return train_loader, val_loader, test_loader
    
    def model(self, train_loader, val_loader, tl_model_ex=False, tl_model='resnet'):
        if tl_model_ex:
            model = Model(2, 128).tl_model()
        else:
            model = Model(2, 128)
        model = model.to(self.device)
        train = Train(model, self.number_of_epochs, self.device)
        train.compile()
        trained_model = train.train(train_loader, val_loader)

        return trained_model








if __name__ == '__main__':
    run = Run('cuda', 128, 1)
    train, val, test = run.dataset()
    model = run.model(train, val, tl_model_ex=True)