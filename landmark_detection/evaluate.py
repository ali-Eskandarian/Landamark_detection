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

        with torch.no_grad():
            self.model.eval()
            for i in range(16):
                try:
                    ex_images, ex_labels = next(examples)
                    plt.subplot(4, 4, i+1)
                    #print(ex_images.shape)
                    image = torch.squeeze(ex_images).permute(1, 2, 0)
                    plt.imshow(image)
                    plt.axis('off')
                    pred = self.model(ex_images.to(self.device))
                    ex_labels, pred = torch.squeeze(ex_labels), torch.squeeze(pred)
                    labels_x, labels_y = [ex_labels[0].item()] + [ex_labels[2].item()], [ex_labels[1].item()] + [ex_labels[3].item()]
                    preds_x, preds_y = [pred[0].item()] + [pred[2].item()], [pred[1].item()] + [pred[3].item()]
                    for point_p, point_ex in zip(zip(labels_x, labels_y), zip(preds_x, preds_y)):
                        plt.plot(122*point_p[0], 122*point_p[1], color='red')
                        plt.plot(122*point_ex[0], 122*point_ex[1], color='white')
                    plt.title(f"Euler Angles Predictions: {pred[4:]} vs {ex_labels[4:]}")
                except GeneratorExit:
                    print("Halalet")


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