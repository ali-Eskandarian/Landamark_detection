import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision
import os
import numpy as np

class Dataset_lip(Dataset):
    def __init__(self, csv_file : str, root_dir : str, transform=None):
        """
        csv_file : path to the csv file containing the training examples data
        root_dir : root directory or main folder path
        transfrom : any specific transform function needed to apply on image
        """
        a = pd.read_csv(csv_file)

        self.annotations = pd.read_csv(csv_file)[['Jaw x', ' Jaw y', ' Nose x', ' Nose y', ' pitch', ' yaw', ' roll',
       ' masked', ' Image Path']]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        img_path = self.root_dir + self.annotations.iloc[index, 8][1:]                               # get the image path
        img_path = img_path.replace("\\", "/")                                                      # replace backslashes with forward slashes
        image = torchvision.io.read_image(img_path)                                                 # read image
        label = torch.from_numpy(self.annotations.iloc[index, :8].values.astype(np.float32))
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_len(self):
        return len(self.annotations)


if __name__ == "__main__":
    main_root = os.getcwd()
    data_train = Dataset_lip(os.path.join(main_root, "Train_Images_f_8.csv"),
                             os.path.join(main_root, "train_data-Copy-Copy"))
    train_loader = DataLoader(dataset=data_train, batch_size=1, shuffle=True)
