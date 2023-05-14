import os
from train import Train
from load_data import Dataset_lip
from torch.utils.data import DataLoader
import torch


class landmark_detection:
    def __init__(self, device, batch_size, number_of_epochs, detector, tl_model_ex=False):
        self.tl_model_ex = tl_model_ex
        self.device = device
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.detector = detector

    def _initialize_(self):
        if self.tl_model_ex:
            self.detector = self.detector(16, self.batch_size).tl_model()
        else:
            self.detector = self.detector(16, self.batch_size)

        return self.detector

    def _dataset_(self):
        main_folder_path = os.getcwd()
        data = Dataset_lip(r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\train_data\Train_Images_f_1.csv",
                           r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\train_data")
        train_set, val_set = torch.utils.data.random_split(data, [int(0.8*data.get_len()),
                                                                  int(0.2*data.get_len()) + 1])
        train_data_loader = DataLoader(train_set, batch_size=self.batch_size, drop_last=True)
        val_data_loader = DataLoader(val_set, batch_size=self.batch_size, drop_last=True)

        return train_data_loader, val_data_loader

    def training(self, save_root, tl_model='resnet'):
        train_loader, val_loader = self._dataset_()
        self.detector = self._initialize_().to(self.device)
        trainer = Train(self.detector, self.number_of_epochs, save_root, self.device)
        trainer.initialize(loss_f="MSE")
        trained_model = trainer.train(train_loader, val_loader)
        trainer.plot_loss(trainer.val_losses, "Validation total Loss", "Val Loss")
        trainer.plot_loss(trainer.val_losses_angles, "Validation angle Loss", "Loss")
        trainer.plot_loss(trainer.val_losses_positions, "Validation position Loss", "Loss")
        trainer.plot_loss(trainer.losses, "Train total Loss", "Train Loss")
        trainer.plot_loss(trainer.losses_angles, "Train angle Loss", "Loss")
        trainer.plot_loss(trainer.losses_positions, "Train position Loss", "Loss")
        return trained_model


