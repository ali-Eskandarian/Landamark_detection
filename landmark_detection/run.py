import os
from model import landmarks_angles_detector
from train import Train
from load_data import Dataset_lip
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class landmark_detection:
    def __init__(self, device, batch_size, number_of_epochs, detector):
        self.device = device
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.detector = detector

    def _dataset(self):
        main_folder_path = os.getcwd()
        data = Dataset_lip(r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\train_data\Train_Images_f_1.csv",
                           r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\train_data")
        data_loader = DataLoader(data, batch_size=self.batch_size, drop_last=True)

        return data_loader

    def training(self, save_root, tl_model_ex=False, tl_model='resnet'):
        train_loader = self._dataset()
        if tl_model_ex:
            self.detector = self.detector(2, self.batch_size).tl_model()
        else:
            self.detector = self.detector(2, self.batch_size)
        self.detector = self.detector.to(self.device)
        trainer = Train(self.detector, self.number_of_epochs, save_root, self.device)
        trainer.compile(loss_f="MSE")
        trained_model = trainer.train(train_loader)
        trainer.plot_loss()
        return trained_model


if __name__ == '__main__':
    run = landmark_detection('cuda', 128, 30, landmarks_angles_detector)
    model = run.training(save_root=r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\model")
