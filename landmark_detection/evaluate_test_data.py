import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import landmarks_angles_detector
from load_data import Dataset_lip

if __name__ == "__main__":
    test_dataset = Dataset_lip(r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\test_data\Test_Images_f_1.csv",
                               r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\test_data")
    test_loader = DataLoader(test_dataset, 16, True, drop_last=True)
    iter_test = iter(test_loader)
    images, labels = iter_test.next()
    model = landmarks_angles_detector(2, 16)
    model.load_state_dict(torch.load(r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\model\model.pt"))
    plt.figure(figsize=(20, 20))
    for index, (image, label) in enumerate(zip(images, labels)):
        pred = model(torch.unsqueeze(image / 255, 0))
        plt.subplot(4, 4, index + 1)
        plt.imshow(image.permute(2, 1, 0))
        # plt.imshow(image.permute(1, 2, 0))
        pred = pred.squeeze()
        plt.plot(122 * label[1].item(), 112 * label[0].item(), marker="o", color='red')
        plt.plot(122 * label[3].item(), 112 * label[2].item(), color='red', marker="o")
        plt.plot(122 * pred[1].item(), 112 * pred[0].item(), marker="v", color='blue')
        plt.plot(122 * pred[3].item(), 112 * pred[2].item(), color='blue', marker="v")
        pred_angles = [(round(360 * (angle.item() - 0.5), 2)) for angle in pred[4:]]
        label_angles = [(round(360 * (angle.item() - 0.5), 2)) for angle in label[4:]]
        plt.axis('off')
        plt.title(f"Predicted Euler angles vs ground truth angles: \n {pred_angles} vs {label_angles}", fontsize=10)
    plt.show()
