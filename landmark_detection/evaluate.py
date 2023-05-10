import matplotlib.pyplot as plt
import torch
import seaborn as sns

class Evaluate():

    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def show_test_results(self, test_loader):

        plt.figure(figsize=(20, 20))
        examples = iter(test_loader)
        ex_images, ex_labels = next(examples)

        with torch.no_grad():
            self.model.eval()
            for i in range(16):
                plt.subplot(4, 4, i+1)
                image = torch.squeeze(ex_images[i]).permute(1, 2, 0)
                plt.imshow(image)
                plt.axis('off')
                pred = self.model(torch.unsqueeze(ex_images[i].to(self.device), 0))
                print(ex_labels)
                labels_x, labels_y = list(ex_labels[0]) + list(ex_labels[2]), list(ex_labels[1]) + list(ex_labels[3])
                preds_x, preds_y = list(pred[0]) + list(pred[2]), list(pred[1]) + list(pred[3])
                for point_p, point_ex in zip(zip(labels_x, labels_y), zip(preds_x, preds_y)):
                    plt.plot(122*point_p[0], 122*point_p[1], color='red')
                    plt.plot(122*point_ex[0], 122*point_ex[1], color='white')
                plt.title(f"Euler Angles Predictions: {pred[4:]} vs {ex_labels[4:]}")

    def confusion_matrix(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for (images, labels) in test_loader:
                preds = self.model(images.to(self.device))
                preds = torch.round(preds)
                preds = preds.squeeze_()
                tp += sum(preds*labels)
                tn += sum(preds==labels) - sum(preds*labels)
                fp += sum(preds) - sum(preds*labels)
                fn += sum(labels) - sum(preds*labels)

            precision = tp/(tp + fp)
            recall = tp/(tp + fn)
            F1score = 2*precision*recall/(precision + recall)
            cm = [[tp, fp], [fn, tn]]
            print(f"The F1 Score for this data is {F1score}")
            sns.heatmap(cm, cmap='Blues', annot=True)