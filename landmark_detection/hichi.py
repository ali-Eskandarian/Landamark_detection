labels_path = "train_data/list3.txt"
images_path = "train_data"
import numpy as np
file_list = [["Jaw x", "Jaw y", "Nose x", "Nose y", "Alpha", "Beta", "Gama", "Image Path"]]
with open(labels_path, 'r') as list_of_labels:
    list_of_labels = list_of_labels.read().split("\n")
    for labels in list_of_labels:
        sub_labels = labels.split(" ")
        label_name = sub_labels[0].split("\\")[-1]
        angles = [float(point)/360+0.5 for point in sub_labels[203:206]]
        njea = [str(point) for point in sub_labels[108:110] + sub_labels[32:34] + angles]
        #njea = torch.tensor(njea, dtype=torch.float32)
        #image_path = images_path + "/imgs_masked/" + label_name[:-4] + "_surgical.png"
        image_path = images_path + "/imgs/" + label_name
        njea.append(image_path)
        file_list.append(njea)
file_list = np.array(file_list)
np.savetxt("Train_Images_normal.csv", file_list, delimiter=", ", fmt="% s")


"""import pandas as pd
from PIL import Image

data = pd.read_csv("Train_Images.csv")
delete_rows = []
for i in range(len(data)):
    try:
        Image.open(data.iloc[i, 7][1:])
    
    except FileNotFoundError:
        delete_rows.append(i)

new_data = data.drop(delete_rows, axis=0)
new_data.to_csv("Clean_Train_Images.csv", index=False)"""