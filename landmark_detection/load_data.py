import torch
from torch.utils.data import IterableDataset, DataLoader
import PIL
from torchvision import transforms
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from skimage import io
import numpy as np


# class Load_data(IterableDataset):
#     def __init__(self, labels_path, images_path):
#
#         self.labels_path = labels_path                                                                      # directory of the labels
#         self.images_path = images_path                                                                      # directory of the images
#
#     def parse_file(self):                                                                                   # function to parse the example from data loader
#
#         with open(self.labels_path, 'r') as list_of_labels:                                                 # open the labels path to extract the labels value
#
#             list_of_labels = list_of_labels.read().split("\n")                                              # split the list of labels into a list of lines that represents the labels of each example
#
#             for labels in list_of_labels:                                                                   # iterate over each label
#                 sub_labels = labels.split(" ")                                                              # split the label elements
#                 label_name = sub_labels[0].split("\\")[-1]                                                  # extract the image name from the first element of the label
#                 angles = [float(point)/360+0.5 for point in sub_labels[203:206]]                            # extract the image angles from the last three elements of the label, divided by 360 plus 0.5 in order to be scaled between 0 and 1
#                 njea = [float(point) for point in sub_labels[108:110] + sub_labels[32:34] + angles]         # extract the nose and jaw positions, save the with angels as a list to represent the image label
#                 njea = torch.tensor(njea, dtype=torch.float32)
#                 print(self.images_path)# convert the label list to tensor
#                 image_path = self.images_path + rf"\imgs_masked\{label_name[:-4]}_surgical.png"             # create image path
#
#                 try:
#                     image = PIL.Image.open(image_path)                                                      # open the image path
#                     image_tensor = transforms.ToTensor()(image)                                             # convert the image to tensor
#                     yield image_tensor,  njea                                                               # return the image and the label
#
#                 except FileNotFoundError:                                                                   # there are some files that are missing from masked images but there is label in the labels list
#                     pass
#
#     def __iter__(self):
#         return self.parse_file()Jaw x	 Jaw y	 Nose x	 Nose y	pitch	 yaw	 roll	Image Path

class Dataset_lip(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        x = pd.read_csv(csv_file)
        self.annotations = pd.read_csv(csv_file)[["Jaw x", "Jaw y",
                                                  "Nose x", "Nose y",
                                                  "pitch", "yaw", "roll", "Image Path"]]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.root_dir + self.annotations.iloc[index, 7]
        image = torch.from_numpy(io.imread(img_path)).T
        a = self.annotations.iloc[index, :7].values
        a = a.astype(np.float)
        y_label = torch.tensor(a)
        if self.transform:
            image = self.transform(image)

        return image, y_label


data_train = Dataset_lip(r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\train_data\Train_Images_f_1.csv",
                         r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\train_data")
train_loader = DataLoader(dataset=data_train, batch_size=1, shuffle=True)
