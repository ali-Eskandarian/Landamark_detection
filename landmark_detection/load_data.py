import torch
from torch.utils.data import IterableDataset, DataLoader
import PIL
from torchvision import transforms
class Load_data(IterableDataset):
    
    def __init__(self, labels_path, images_path):
        
        self.labels_path = labels_path                                                                      # directory of the labels
        self.images_path = images_path                                                                      # directory of the images

    def parse_file(self):                                                                                   # function to parse the example from data loader
        
        with open(self.labels_path, 'r') as list_of_labels:                                                 # open the labels path to extract the labels value
            
            list_of_labels = list_of_labels.read().split("\n")                                              # split the list of labels into a list of lines that represents the labels of each example
            
            for labels in list_of_labels:                                                                   # iterate over each label
                sub_labels = labels.split(" ")                                                              # split the label elements
                label_name = sub_labels[0].split("\\")[-1]                                                  # extract the image name from the first element of the label
                angles = [float(point)/360+0.5 for point in sub_labels[203:206]]                            # extract the image angles from the last three elements of the label, divided by 360 plus 0.5 in order to be scaled between 0 and 1
                njea = [float(point) for point in sub_labels[108:110] + sub_labels[32:34] + angles]         # extract the nose and jaw positions, save the with angels as a list to represent the image label
                njea = torch.tensor(njea, dtype=torch.float32)                                              # convert the label list to tensor
                image_path = self.images_path + "/imgs_masked/" + label_name[:-4] + "_surgical.png"         # create image path
                
                try:    
                    image = PIL.Image.open(image_path)                                                      # open the image path
                    image_tensor = transforms.ToTensor()(image)                                             # convert the image to tensor
                    yield image_tensor,  njea                                                               # return the image and the label
                
                except FileNotFoundError:                                                                   # there are some files that are missing from masked images but there is label in the labels list
                    pass

    def __iter__(self):
        return self.parse_file()