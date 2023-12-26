import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import landmarks_angles_detector
from load_data import Dataset_lip

# class Evaluate():
#     def __init__(self, data):
#         self.data = data
#
#     def show_test_results(self):
#         model = landmarks_angles_detector(16, 16, previous_model=False)
#         model.load_state_dict(torch.load("model.pt"))
#
#         plt.figure(figsize=(20, 20))
#         for index, (images, labels) in enumerate(self.data, start=1):  # Start enumerate from index 1
#             image = images[0]  # Assuming you have only one image in each batch
#             image = image.permute(1, 2, 0)
#
#             pred = model(torch.unsqueeze(images[0] / 255, 0))
#             pred = pred.squeeze()
#
#             plt.subplot(4, 4, index)  # Update the subplot index
#
#             plt.imshow(image[int(112 * pred[0].item()):int(112 * pred[2].item()), :, :])
#             plt.axis('off')
#             if index == 16:
#                 break
#         plt.show()
#         plt.savefig("a.png")
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import landmarks_angles_detector
from load_data import Dataset_lip
a, b  = 50, 35
class Evaluate():
    def __init__(self, data):
        self.data = data

    def show_test_results(self):
        model = landmarks_angles_detector(16, 16, previous_model=False)
        model.load_state_dict(torch.load("model.pt"))


        for index, (images, labels) in enumerate(self.data, start=1):  # Start enumerate from index 1
            image = images[0]  # Assuming you have only one image in each batch
            labels = labels[0, :-1] *112 # Assuming you have only one image in each batch
            image = image.permute(1, 2, 0)

            pred = model(torch.unsqueeze(images[0] / 255, 0))
            pred = pred.squeeze()*112

            # plt.subplot(4, 4, index)  # Update the subplot index
            plt.figure(figsize=(20, 20))

            # plt.axis('off')

            # if index in [3, 6, 9, 12]:
            #     plt.subplots_adjust(hspace=0.5)  # Increase vertical spacing between subplots
            if index < 5 * a :
                plt.imshow(image)
                # Plot labels (nose and cheek) in green
                plt.plot(labels[1], labels[0], marker="o", color='green', markersize=b)  # Nose point
                plt.plot(labels[3], labels[2], marker="o", color='green', markersize=b)  # Cheek point

                # Plot predictions in red
                plt.plot(pred[1].item(), pred[0].item(), marker="o", color='red', markersize=b)  # Predicted nose point
                plt.plot(pred[3].item(), pred[2].item(), marker="o", color='red', markersize=b)  # Predicted cheek point
                plt.grid(True, color='k',linewidth=0.9)
                plt.savefig(f"save/grid/b_{index}.png")
            elif index <10 * a:
                plt.imshow(image)
                # Plot labels (nose and cheek) in green
                plt.plot(labels[1], labels[0], marker="o", color='green', markersize=b)  # Nose point
                plt.plot(labels[3], labels[2], marker="o", color='green', markersize=b)  # Cheek point

                # Plot predictions in red
                plt.plot(pred[1].item(), pred[0].item(), marker="o", color='red', markersize=b)  # Predicted nose point
                plt.plot(pred[3].item(), pred[2].item(), marker="o", color='red', markersize=b)  # Predicted cheek point

                plt.savefig(f"save/gtvspred/b_{index-50}.png")
            elif index >10 * a:
                plt.imshow(image[int(pred[0].item()):int(pred[2].item()), :, :])
                plt.axis('off')
                plt.savefig(f"save/lip/b_{index-100}.png")
            if index == 15*a:
                break
            plt.close()

if __name__ == "__main__":
    test_dataset = Dataset_lip("Train_Images_f_8.csv",
                                "train_data-Copy-Copy")
    test_loader = DataLoader(test_dataset, 1, True, drop_last=True)
    eval = Evaluate(test_loader)
    eval.show_test_results()
