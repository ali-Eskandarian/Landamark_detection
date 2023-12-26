import os
from train import Train
from load_data import Dataset_lip
from torch.utils.data import DataLoader
import torch


class landmark_detection:
    
    def __init__(self, device : str, batch_size : int, number_of_epochs : int, detector, tl_model_ex=False, load=False):
        
        """
        device :the device which the model is trained on
        batch_size : the number of training examples in each batch
        number_of_epochs : the number of training epochs
        detector : the model to be trained
        tl_model_ex : whether to use pretrained transfer learning model
        load : whether to load pretrained model
        """
        
        self.tl_model_ex = tl_model_ex
        self.device = device
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.detector = detector
        self.load = load

    def _initialize_(self):
        
        if self.load:                                                                                   # load the pretrained model
            print("pre_load model detected")
        elif self.tl_model_ex:                                                                          # load the pretrained transfer learning model
            self.detector = self.detector(2, self.batch_size).tl_model()
        else:                                                                                           # load the simple model from scratch
            self.detector = self.detector(2, self.batch_size, True)

        return self.detector

    def _dataset_(self):
        
        main_folder_path = os.getcwd()                                                                  # get the current working directory
        data = Dataset_lip(os.path.join(main_folder_path, "Train_Images_f_8.csv"),
                           os.path.join(main_folder_path, "train_data-Copy-Copy"))                                # get the main dataset
        train_set, val_set = torch.utils.data.random_split(data, [int(0.8*data.get_len()),
                                                                  int(0.2*data.get_len()) + 1])         # split the data into training and validation sets
        train_data_loader = DataLoader(train_set, batch_size=self.batch_size, drop_last=True)
        val_data_loader = DataLoader(val_set, batch_size=self.batch_size, drop_last=True)

        return train_data_loader, val_data_loader

    def training(self, save_root : str):
        
        train_loader, val_loader = self._dataset_()
        # self.detector = self._initialize_().to(self.device)
        self.detector = self.detector.to(self.device)

        trainer = Train(self.detector, self.number_of_epochs, save_root, self.device)
        trainer.initialize(loss_f="custom_2", learning_rate=0.001)
        trained_model = trainer.train(train_loader, val_loader)
        
        trainer.plot_loss(trainer.losses["Validation Loss"], "Validation total Loss", "Val Loss")
        trainer.plot_loss(trainer.losses["Validation Loss Angles"], "Validation angle Loss", "Loss")
        trainer.plot_loss(trainer.losses["Validation Loss Positions"], "Validation position Loss", "Loss")
        trainer.plot_loss(trainer.losses["Training Loss"], "Train total Loss", "Train Loss")
        trainer.plot_loss(trainer.losses["Training Loss Angles"], "Train angle Loss", "Loss")
        trainer.plot_loss(trainer.losses["Training Loss Positions"], "Train position Loss", "Loss")
        
        return trained_model


