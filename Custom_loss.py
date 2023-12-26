import torch.nn as nn
import torch
import math
import numpy as np

def euler_to_vector(euler_angles : torch):
    # Compute the rotation matrix
    # create a tensor representing the z-axis in the frame of the euler angles
    z_axis = torch.tensor([0, 0, 1], dtype=torch.float32).to("cuda")

    # convert euler angles to rotation matrices
    batch_size = euler_angles.shape[0]
    pitch, yaw, roll = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    c1, c2, c3 = torch.cos(pitch), torch.cos(yaw), torch.cos(roll)
    s1, s2, s3 = torch.sin(pitch), torch.sin(yaw), torch.sin(roll)
    r11, r12, r13 = c2 * c3, -c2 * s3, s2
    r21, r22, r23 = c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1
    r31, r32, r33 = s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2

    r = torch.stack([torch.stack([r11, r12, r13], dim=1),
                     torch.stack([r21, r22, r23], dim=1),
                     torch.stack([r31, r32, r33], dim=1)], dim=1)

    # multiply rotation matrices with the z-axis vector
    vectors = torch.matmul(r, z_axis)

    # normalize the vectors to get unit vectors
    unit_vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

    # reshape the unit vectors to the desired shape
    unit_vectors = unit_vectors.view(batch_size, 3)

    return unit_vectors


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    @staticmethod
    def forward(output, target):
        
        torch.pi = math.pi
        output_a, output_p = 2 * torch.pi * (output[:, 4:] - 0.5), output[:, :4]                    # convert degrees to radians
        target_a, target_p = 2 * torch.pi * (target[:, 4:] - 0.5), target[:, :4]
        vec_o, vec_t = euler_to_vector(output_a), euler_to_vector(target_a)                         # calculate euler angles by degrees
        L_a = torch.norm(torch.cross(vec_o, vec_t, dim=1)) + (1 - (vec_o * vec_t).sum(dim=1))       # compute loss by calculating the norm of cross product and inner product
        L_p = torch.sum( (output_p - target_p) * (output_p - target_p), axis=1)                     # calculate the squared error
        loss = torch.mean(L_p + L_a)                                                                # total loss is equal to the mean of the squared error and product error

        return loss, torch.mean(L_a), torch.mean(L_p)

class CLoss(nn.Module):
    def __init__(self):
        super(CLoss, self).__init__()

    @staticmethod
    def forward(output, target):
        
        diff = output - target                                                                      # compute difference between output and target
        norm = 0
        
        for element in diff:
            norm += element**2                                                                      # calculate square of difference
        
        norm = torch.pow(norm, 1/3)
        loss = torch.sum(-torch.log10(1 - norm))                                                    # compute loss by calculating sum of log of norm
        
        return loss
