import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from functools import reduce
from operator import __add__


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
                                             [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, x):
        return self._conv_forward(self.zero_pad_2d(x), self.weight, self.bias)

class Model(nn.Module):
  
  def __init__(self, initial_filter_size, batch_size):
    
    super(Model, self).__init__()
    self.batch_size = batch_size
    self.conv1 = Conv2dSamePadding(3, initial_filter_size, 11)
    self.batchn1 = nn.BatchNorm2d(initial_filter_size)
    self.pool = nn.MaxPool2d(2)
    self.conv2 = Conv2dSamePadding(initial_filter_size, 2*initial_filter_size, 7)
    self.batchn2 = nn.BatchNorm2d(2*initial_filter_size)
    self.conv3 = Conv2dSamePadding(2*initial_filter_size, 4*initial_filter_size, 5)
    self.batchn3 = nn.BatchNorm2d(4*initial_filter_size)
    self.conv4 = Conv2dSamePadding(4*initial_filter_size, 8*initial_filter_size, 3)
    self.batchn4 = nn.BatchNorm2d(8*initial_filter_size)
    self.lin1 = nn.Linear(7*7*8*initial_filter_size, 64)
    self.lin2 = nn.Linear(64, 7)

  def forward(self, x):
    
    x = F.relu(self.conv1(x))
    x = self.batchn1(x)
    x = self.pool(x)

    x = F.relu(self.conv2(x))
    x = self.batchn2(x)
    x = self.pool(x)

    x = F.relu(self.conv3(x))
    x = self.batchn3(x)
    x = self.pool(x)

    x = F.relu(self.conv4(x))
    x = self.batchn4(x)
    x = self.pool(x)
    
    batch = x.shape[0]
    x = x.view([batch, -1])

    x = F.relu(self.lin1(x))
    out = torch.clone(F.sigmoid(self.lin2(x)))

    return out
  
  def tl_model(self, main_model="resnet"):
    
    if main_model == "resnet":
        main_model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

    for param in main_model.parameters():
        param.requires_grad = False

    num_out_features = main_model.fc.in_features
    main_model.fc = nn.Linear(num_out_features, 7)

    return main_model