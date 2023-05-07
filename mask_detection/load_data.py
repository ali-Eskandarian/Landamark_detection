import torch
import torchvision
import torchvision.transforms as tvtf
from torch.utils.data import DataLoader

class Load_data():

    def __init__(self, device : str, main_folder_path : str, train_folder_name : str, test_folder_name : str):
        
        if train_folder_name[0] == "/":
            train_folder_name = train_folder_name[1:]

        if test_folder_name[0] == "/":
            test_folder_name = test_folder_name[1:]        
        self.device = device
        self.main_path = main_folder_path
        self.train_path = train_folder_name
        self.test_path = test_folder_name
    
    def __read_data(self, train_val_split_ratio):

        assert train_val_split_ratio < 1 and train_val_split_ratio > 0 , "Please enter a number between 0 and 1"
        
        device = torch.device(self.device)
        main_dataset = torchvision.datasets.ImageFolder(self.main_path + "/" + self.train_path, transform=tvtf.ToTensor(), target_transform=Load_data.__convert_to_float)
        train_dataset, val_dataset = Load_data.__split_train_val(main_dataset, train_val_split_ratio)
        test_dataset = torchvision.datasets.ImageFolder(self.main_path + "/" + self.test_path, transform=tvtf.ToTensor(), target_transform=Load_data.__convert_to_float)

        return train_dataset, val_dataset, test_dataset
    
    def __split_train_val(dataset, split_ratio):

        number_of_samples = len(dataset)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(number_of_samples*split_ratio), number_of_samples-int(number_of_samples*split_ratio)])

        return train_dataset, val_dataset
    
    def __convert_to_float(value):
        
        return float(value)
    
    def __load_dataset(dataset, batch_size, shuffle=True, drop_last=True):
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        return dataloader
    
    def load(self, batch_size : int, train_val_split_ratio : float, shuffle_test=True):

        train, val, test = self.__read_data(train_val_split_ratio)
        train_loader = Load_data.__load_dataset(train, batch_size)
        val_loader = Load_data.__load_dataset(val, batch_size)
        test_loader = Load_data.__load_dataset(test, batch_size, shuffle=shuffle_test)

        return train_loader, val_loader, test_loader