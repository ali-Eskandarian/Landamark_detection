import torch.nn as nn
import torch
import math
import numpy as np


def euler_to_vector(yaw, pitch, roll):
    # Compute the rotation matrix
    R_yaw = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                          [torch.sin(yaw), torch.cos(yaw), 0],
                          [0, 0, 1]])
    R_pitch = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                            [0, 1, 0],
                            [-torch.sin(pitch), 0, torch.cos(pitch)]])
    R_roll = torch.tensor([[1, 0, 0],
                           [0, torch.cos(roll), -torch.sin(roll)],
                           [0, torch.sin(roll), torch.cos(roll)]])
    R = torch.matmul(torch.matmul(R_yaw, R_pitch), R_roll)

    # Compute the vector in the new orientation
    v = torch.matmul(R, torch.tensor([0, 0, 1], dtype=torch.float32))

    return v


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    @staticmethod
    def forward(output, target):
        loss = 0
        torch.pi = math.pi
        output = 2 * torch.pi * (output - 0.5)
        target = 2 * torch.pi * (target - 0.5)
        for o, t in zip(output, target):
            vec_o = euler_to_vector(o[0], o[1], o[2])
            vec_t = euler_to_vector(t[0], t[1], t[2])
            loss += 1 - torch.norm(torch.cross(vec_o, vec_t)) + torch.dot(vec_o, vec_t)
        return loss
