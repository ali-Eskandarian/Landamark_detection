import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
from model import Model

class Load_test_data(Dataset):

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, idx):
        
        image_path = self.data.iloc[idx][7]
        image = read_image(image_path[1:])
        label = self.data.iloc[idx][:7]
        return image, torch.tensor(label)


if __name__ == "__main__":
    test_dataset = Load_test_data("Clean_Test_Images.csv")
    test_loader = DataLoader(test_dataset, 16, True, drop_last=True)
    iter_test = iter(test_loader)    
    images, labels = iter_test.next()
    model = Model(2, 16)
    model.load_state_dict(torch.load("model/landmark_detection/model.pt"))
    plt.figure(figsize=(20, 20))
    for index, (image, label) in enumerate(zip(images, labels)):
        pred = model(torch.unsqueeze(image/255, 0))
        plt.subplot(4, 4, index+1)
        plt.imshow(image.permute(1, 2, 0))
        pred = pred.squeeze()
        plt.plot(122*label[1].item(), 122*label[0].item(), marker="o", color='red')
        plt.plot(122*label[3].item(), 122*label[2].item(), color='red', marker="o")
        plt.plot(122*pred[1].item(), 122*pred[0].item(), marker="v", color='blue')
        plt.plot(122*pred[3].item(), 122*pred[2].item(), color='blue', marker="v")
        pred_angles = [(round(180*(angle.item() - 0.5), 2)) for angle in pred[4:]]
        label_angles = [(round(180*(angle.item() - 0.5), 2)) for angle in label[4:]]
        plt.axis('off')
        plt.title(f"Predicted Euler angles vs ground truth angles: \n {pred_angles} vs {label_angles}", fontsize=10)
    plt.show()