"""Contains the pytorch neural network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class gnet(nn.Module):
  def __init__(self, initbal=0):
    super(gnet, self).__init__()
    self.balance = initbal
    self.conv1 = nn.Conv2d(1, 600, 3)
    self.conv2 = nn.Conv2d(600, 1200, 2)
    self.conv3 = nn.Conv2d(1200, 2100, 2)
      # GRU layer
    self.gru = nn.GRU(2100, 1200, 2)
      # Linear  
    self.fc1 = nn.Linear(1200, 900) 
    self.fc2 = nn.Linear(900, 600)
      # Value head
    self.fc3 = nn.Linear(600, 300)
    self.fc4 = nn.Linear(300, 1)
      # Mean head
    self.fc5 = nn.Linear(600, 300)
    self.fc6 = nn.Linear(300, 7)
      # Varinace head
    self.fc7 = nn.Linear(600, 300)
    self.fc8 = nn.Linear(300, 7)

  def num_flat_features(self, x):
      size = x.size()[1:]
      num_features = 1
      for s in size:
        num_features *= s
      return num_features

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (3,3))
    x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
    x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

    x = x.view(-1, self.num_flat_features(x))
    #x = torch.flatten(x, 1)
    x = x.unsqueeze(-2)
    h_n = torch.zeros(2, x.size(0), 1200)
        # gru output
    gru_out, _ = self.gru(x, h_n)
    x = F.relu(self.fc1(gru_out))         
    x = F.relu(self.fc2(x))          
        # Value output
    Value = F.relu(self.fc3(x))
    Value = self.fc4(Value)
        # Mean output
    Mean = F.relu(self.fc5(x))
    Mean = F.relu(self.fc6(Mean))
        # Variance output
    Variance = F.relu(self.fc7(x))
    Variance = F.relu(self.fc8(Variance)) 
                  
    return Value, Mean, Variance
