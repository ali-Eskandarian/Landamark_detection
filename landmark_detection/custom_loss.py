import torch.nn as nn
import torch
import math


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()


    def forward(self, output, target):
        loss = 0
        torch.pi = math.pi
        output = torch.cos(2 * torch.pi * (output - 0.5))
        target = torch.cos(2 * torch.pi * (target - 0.5))
        for vec_o, vec_t in zip(output, target):
            outer_product = torch.outer(vec_o, vec_t)
            loss += torch.norm(outer_product)
        return loss
