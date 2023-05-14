import os
from model import Model
from train import Train
from load_data import Dataset_lip
from torch.utils.data import DataLoader

class Run():
    def __init__(self, device, batch_size, number_of_epochs):
        self.device = device
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

    def dataset(self):
        main_folder_path = os.getcwd()
        data = Dataset_lip(r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\train_data\Train_Images_f_1.csv",
                            r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\train_data")
        data_loader = DataLoader(data, batch_size=self.batch_size, drop_last=True)

        return data_loader
    
    def model(self, train_loader, tl_model_ex=False, tl_model='resnet'):
        if tl_model_ex:
            model = Model(2, self.batch_size).tl_model()
        else:
            model = Model(2, self.batch_size)
        model = model.to(self.device)
        train = Train(model, self.number_of_epochs, self.device)
        train.compile(loss_f="MSE")
        trained_model = train.train(train_loader)
        train.save_model("model/landmark_detection/model.pt")

        return trained_model

if __name__ == '__main__':
    run = Run('cuda', 16, 30)
    train = run.dataset()
    model = run.model(train)