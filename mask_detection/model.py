import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Model(nn.Module):
  
  def __init__(self, initial_filter_size, batch_size):
    
    super(Model, self).__init__()
    self.batch_size = batch_size
    self.conv1 = nn.Conv2d(3, initial_filter_size, 5, padding='same')
    self.batchn1 = nn.BatchNorm2d(initial_filter_size)
    self.pool = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(initial_filter_size, 2*initial_filter_size, 5, padding='same')
    self.batchn2 = nn.BatchNorm2d(2*initial_filter_size)
    self.conv3 = nn.Conv2d(2*initial_filter_size, 4*initial_filter_size, 3, padding='same')
    self.batchn3 = nn.BatchNorm2d(4*initial_filter_size)
    self.conv4 = nn.Conv2d(4*initial_filter_size, 8*initial_filter_size, 3, padding='same')
    self.batchn4 = nn.BatchNorm2d(8*initial_filter_size)
    self.lin1 = nn.Linear(7*7*8*initial_filter_size, 64)
    self.lin2 = nn.Linear(64, 1)

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
    x = self.lin2(x)

    return x
  
  def tl_model(self, main_model="resnet"):
    
    if main_model == "resnet":
      main_model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

    for param in main_model.parameters():
        param.requires_grad = False

    num_out_features = main_model.fc.in_features
    main_model.fc = nn.Linear(num_out_features, 1)

    return main_model