import torch
from torch.utils.data import IterableDataset, DataLoader
import PIL
from torchvision import transforms
class Load_data(IterableDataset):
    def __init__(self, labels_path, images_path):
        self.labels_path = labels_path
        self.images_path = images_path

    def parse_file(self):
        
        with open(self.labels_path, 'r') as list_of_labels:
            list_of_labels = list_of_labels.read().split("\n")
            for labels in list_of_labels:
                sub_labels = labels.split(" ")
                label_name = sub_labels[0].split("\\")[-1]
                angles = [float(point)/360+0.5 for point in sub_labels[203:206]]
                njea = [float(point) for point in sub_labels[108:110] + sub_labels[32:34] + angles]
                njea = torch.tensor(njea, dtype=torch.float32)
                image_path = self.images_path + "/imgs_masked/" + label_name[:-4] + "_surgical.png"
                try:    
                    image = PIL.Image.open(image_path)
                    image_tensor = transforms.ToTensor()(image)
                    yield image_tensor,  njea
                except:
                    pass

    def __iter__(self):
        return self.parse_file()
    
if __name__== "__main__":
    load = Load_data("train_data/list3.txt", "train_data")
    data_loader = DataLoader(load, batch_size=16, drop_last=True)
    for image, label in data_loader:
        print(image)