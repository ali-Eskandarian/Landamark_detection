from model import landmarks_angles_detector
from detection import landmark_detection


def main():
    run = landmark_detection('cuda', 16, 50, landmarks_angles_detector)
    model = run.training(save_root=r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\model")


if __name__ == '__main__':
    main()
