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
                plt.title(f"Prediction: {pred.item()} vs {ex_labels[i]}")

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