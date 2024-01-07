from model import landmarks_angles_detector
from detection import landmark_detection
from torchsummary import summary
import torch


def main():
    model = run.training()
    model = landmarks_angles_detector(16, 30, previous_model=False).to('cpu')
    summary(model, (3, 112, 112))
    model.load_state_dict(torch.load("model/landmark_detection/inception/pretrained/model_dynamic.pt"))
    run = landmark_detection('cuda', 16, 30, model, load=False)
    trained_model = run.training(save_root="save")

if __name__ == '__main__':
    main()

