# """Contains the pytorch neural network"""
    
import torch
import torch.nn as nn

class ActionNet(nn.Module):
    def __init__(self, num_outputs:int):
        super(ActionNet, self).__init__()
        self.hidden_dim = 1024
        self.num_outputs = num_outputs
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
        )
        
        self.values = nn.Sequential(
            nn.Linear(128*4*4, self.hidden_dim, bias=True), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 7, bias=False), 
        )

    def forward(self, input_dict, state, seq_lens):
        batch_size, _, _, _ = input_dict["obs"].shape
        CNN_out = self.CNN(input_dict["obs"].float())
        # print("CNN", CNN_out.shape)
        flat = CNN_out.view((batch_size, 128*4*4))
        # print("FLAT", flat.shape)
        values = self.values(flat)
        return values
        # return torch.zeros(batch_size, 7, device="cuda"), []