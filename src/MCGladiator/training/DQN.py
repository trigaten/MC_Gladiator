# """Contains the pytorch neural network"""
    
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class DeepQNet(TorchModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # Pass num_outputs=None into super constructor (so that no action/
        # logits output layer is built).
        # Alternatively, you can pass in num_outputs=[last layer size of
        # config[model][fcnet_hiddens]] AND set no_last_linear=True, but
        # this seems more tedious as you will have to explain users of this
        # class that num_outputs is NOT the size of your Q-output layer.
        """NOTE: I BELIEVE above describes error: https://docs.ray.io/en/latest/rllib/rllib-models.html"""
        super(DeepQNet, self).__init__(
            obs_space, action_space, None, model_config, name
        )
        self.no_last_linear = True
        self.hidden_dim = 1024
        # Now: self.num_outputs contains the last layer's size, which
        # we can use to construct the dueling head (see torch: SlimFC
        # below).

        # Construct advantage head ...
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
        
        self.values = nn.Sequential(
            nn.Linear(128*4*4, self.hidden_dim, bias=True), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 7, bias=False), 
        )

    # def get_q_values(self, input):
    #     base_out = self.CNN(input)
    #     values = self.values(base_out)
    #     return values

    def forward(self, input_dict, state, seq_lens):
        batch_size, _, _, _ = input_dict["obs"].shape
        CNN_out = self.CNN(input_dict["obs"].float())
        # print("CNN", CNN_out.shape)
        flat = CNN_out.view((batch_size, 128*4*4))
        # print("FLAT", flat.shape)
        values = self.values(flat)
        # print("VALUES", values.shape)
        return values, []
        # return torch.zeros(batch_size, 7, device="cuda"), []
