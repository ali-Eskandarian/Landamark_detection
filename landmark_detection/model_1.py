from model import Model
import torch
from evaluate import Evaluate
from load_data import Load_data

model = Model(2, 16)
model.load_state_dict(torch.load("model/landmark_detection/model.pt"))
data = Load_data("test_data/list.txt", "test_data")
data_loader = torch.utils.data.DataLoader(data)

eval = Evaluate(model, 'cpu')
eval.show_test_results(data_loader)