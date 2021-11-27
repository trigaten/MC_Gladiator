"""Contains the pytorch neural network"""
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPO_net(nn.Module):
  def __init__(self):
        super().__init__()

        self.CNN = nn.Sequential(nn.Conv2d(1, 32, 3, stride=3), nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 2, stride=2), nn.MaxPool2d(2),
            nn.Conv2d(64, 1200, 2, stride=2), nn.MaxPool2d(2),
        )

        self.gru = nn.GRU(1200, 1200, 2)
        
        self.Linear = nn.Sequential(nn.Linear(1200, 900, bias=True), 
                                    nn.Linear(900, 600, bias=True))
        # Value
        self.value = nn.Sequential(nn.Linear(600, 300, bias=False),
                                     nn.Linear(300, 1, bias=False))
        # Mean
        self.mean = nn.Sequential(nn.Linear(600, 300, bias=False), nn.Sigmoid(),
                                    nn.Linear(300, 7, bias=False))
        # Varinace
        self.variance = nn.Sequential(nn.Linear(600, 300, bias=False), nn.Sigmoid(), 
                                    nn.Linear(300, 7, bias=False))

  def forward(self, x):
        # CNN output
        CNN_out = self.CNN(x)
        # flatten
        flat = torch.flatten(CNN_out, 1)
        flat = torch.unsqueeze(flat, 0)
        # pass 0s as hidden state
        h_0 = torch.zeros(2, 1, 1200)
        # gru output
        gru_out, h_n= self.gru(flat, (h_0))
        # linear layer output
        lin_out = self.Linear(gru_out)                           
        # Value output
        Value = self.value(lin_out)
        # Mean output
        Mean = self.mean(lin_out)                         
        # Variance output
        Variance = self.variance(lin_out)
                                     
        return Value, Mean, Variance
