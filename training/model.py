"""Contains the pytorch neural network"""
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax

class Discrete_PPO_net(nn.Module):
  """This network is composed of a CNN which feeds into both a Linear
  layer and a GRU. The outputs from these layers are concatenated then fed 
  through two linear heads to compute action probabilities and values. The action
  logits are fed through the softmax to compute the actual probabilities.
  """
  def __init__(self, num_actions):
        super().__init__()
        self.sm = nn.Softmax(dim=2)
        self.num_actions = num_actions

        # maintain spatial resolution with padding
        self.CNN = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3), 
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2)
            # here 128 channels, 4x4 = 2048 units
        )

        self.base_GRU = nn.GRU(128*4*4, 1024)
        
        self.base_fc = nn.Sequential(
          nn.Linear(128*4*4, 1024, bias=True), 
          nn.ReLU()
        )

        # Action
        self.action = nn.Sequential(
          nn.Linear(2048, 512, bias=False), 
          nn.ReLU(),
          nn.Linear(512, num_actions, bias=False),
        )

        # Value
        self.value = nn.Sequential(
          nn.Linear(2048, 512, bias=False),
          nn.ReLU(),
          nn.Linear(512, 1, bias=False),
        )

  def forward(self, x, gru_hidden=None):
    """
    :param x: [N, 1, height, width] pytorch tensor
    """
    batch_size = x.shape[0]
    # CNN output
    CNN_out = self.CNN(x)

    flat = CNN_out.view((1, batch_size, 2048))

    if gru_hidden != None:
      GRU_out, h_n = self.base_GRU(flat, gru_hidden)
    else:
      GRU_out, h_n = self.base_GRU(flat)

    base_fc_out = self.base_fc(flat)

    # [1,1,2048]
    base_out = torch.cat((GRU_out, base_fc_out), dim=2)
              
    # Action output
    action = self.action(base_out)
    action = self.sm(action)

    # Value output
    value = self.value(base_out)                  
                    
    # return action, value, hidden
    return action, value, h_n
