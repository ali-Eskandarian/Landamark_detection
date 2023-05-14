from copy import deepcopy as dc
import pandas as pd
from skimage import io
labels_path = "list3.txt"
images_path = r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\train_data"
import numpy as np
#
# file_list = [["Jaw x", "Jaw y", "Nose x", "Nose y", "pitch", "yaw", "roll", "Image Path"]]
# with open(labels_path, 'r') as list_of_labels:
#     list_of_labels = list_of_labels.read().split("\n")
#     for labels in list_of_labels:
#         sub_labels = labels.split(" ")
#         label_name = sub_labels[0].split(f"\\")[-1]
#         angles = [(float(point) / 360 + 0.5) for point in sub_labels[203:206]]
#         njea = [float(point) for point in sub_labels[108:110] + sub_labels[32:34] + angles]
#         njea_1 = dc(njea)
#         image_path = rf"\imgs\{label_name}"
#         njea.append(image_path)
#         image_path_1 = rf"\imgs_masked\{label_name[:-4]}_surgical.png"
#         njea_1.append(image_path_1)
#         try:
#             a = r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\train_data"
#             b = image_path
#             c = a + b
#             io.imread(c)
#             file_list.append(njea)
#         except:
#             pass
#         try:
#             a = r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\train_data"
#             b = image_path_1
#             c = a + b
#             io.imread(c)
#             file_list.append(njea_1)
#         except:
#             pass
#
#
# # print(file_list)
# file_list = np.array(file_list, dtype=object)
# np.savetxt("Test_Images_f.csv", file_list, delimiter=", ", fmt="% s")
#

df = pd.read_csv("Test_Images_f.csv")
# df["Jaw x"] = df["Jaw x"].apply(lambda x: float(x[1:]))
# print(df["Image Path"][0][:-1])
#
# df["Image Path"] = df["Image Path"].apply(lambda y: y[:-1])
df["Image Path"] = df["Image Path"].apply(lambda y: y[1:])

# df.to_csv("Test_Images_f_1.csv")