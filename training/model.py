"""Contains the pytorch neural network"""
    
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

class Discrete_PPO_net(RecurrentNetwork, nn.Module):
  """This network is composed of a CNN which feeds into both a Linear
  layer and a GRU. The outputs from these layers are concatenated then fed 
  through two linear heads to compute action logits and values. 
  """
  def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    nn.Module.__init__(self)
    super(Discrete_PPO_net, self).__init__(obs_space, action_space, num_outputs, model_config, name)
    self.obs_space = obs_space
    self.action_space = action_space
    self.num_outputs = num_outputs
    self.base_out = None
    self.hidden_dim = 1024

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

    self.base_GRU = nn.GRU(128*4*4, self.hidden_dim, batch_first=True)
    
    self.base_fc = nn.Sequential(
      nn.Linear(128*4*4, self.hidden_dim, bias=True), 
      nn.ReLU()
    )

    # Action
    self.action = nn.Sequential(
      nn.Linear(2048, 512, bias=False), 
      nn.ReLU(),
      nn.Linear(512, self.num_outputs, bias=False),
    )

    # Value
    self.value = nn.Sequential(
      nn.Linear(2048, 512, bias=False),
      nn.ReLU(),
      nn.Linear(512, 1, bias=False),
    )

  def forward_rnn(self, inputs, state, seq_lens):
    """
    :param inputs: [N, L, 3 (channels) * height * width] pytorch tensor
    """
    inputs = inputs.view(inputs.shape[0], inputs.shape[1], 3, 64, 64)
    batch_size, seq_len, channels, height, width = inputs.shape
    # we have a 5d tensor, but conv2d only accepts 4d ones, 
    # so we will "flatten" the first two dims, batch and seq length
    # this is fine because the cnn just extracts features from each
    # frame
    CNN_in = inputs.view(batch_size*seq_len,channels, height, width)
    # CNN output: (batch_size*seq_len, 128, 4, 4)
    CNN_out = self.CNN(CNN_in)
    # now we have batches of sequences of extracted features (batch_size, seq_len, 2048)
    flat = CNN_out.view((batch_size,seq_len, 128*4*4))

    # (batch_size, seq_len, 1024)
    if state != []:
      h_in = torch.unsqueeze(state[0], 0)
      GRU_out, h_n = self.base_GRU(flat, h_in)
    else:
      GRU_out, h_n = self.base_GRU(flat)

    # (batch_size, seq_len, 1024)
    base_fc_out = self.base_fc(flat)
    # (batch_size, seq_len, 2048)
    self.base_out = torch.cat((GRU_out, base_fc_out), dim=2)
    # Action output (batch_size, seq_len, num_outputs)
    action = self.action(self.base_out)    
    action = action.view(batch_size*seq_len,self.num_outputs) 
    # return with "flattened" first 2 dims: (batch_size*seq_len, num_outputs)
    # this is fine bc we are computing loss next
    return action, [torch.squeeze(h_n, 0)]

  def get_initial_state(self):
    # make it on the same device!
    h = [self.base_fc[0].weight.new(1, self.hidden_dim).zero_().squeeze(0)]
    return h

  def value_function(self):
    batch_size, seq_len, size = self.base_out.shape
    return self.value(self.base_out).view(batch_size*seq_len)


# if __name__ == "__main__":
#   import gym.spaces as spaces
#   action_space = spaces.Discrete(2)
#   observation_space = spaces.Box(0, 255, [3, 64, 64])

#   def env_creator(env_config):
#       opponent = Discrete_PPO_net(observation_space, action_space, 3, "model_config", "name")
