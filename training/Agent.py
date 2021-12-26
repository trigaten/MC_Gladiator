import torch
import numpy as np

class Agent():
    """Class which takes observations and returns the selected action, the torch distribution, and the state values"""
    def __init__(self, net, grad=True):
        self.net = net
        self.grad = grad
        self.hiddens = None
        self.num_actions = net.num_actions
    
    def __call__(self, states):
        if self.grad:
            action_distribution, values, hiddens = self.net(states, self.hiddens)
        else:
            with torch.no_grad():
                action_distribution, values, hiddens = self.net(states, self.hiddens)

        action_distribution_np = torch.reshape(action_distribution, (self.num_actions,)).detach().cpu().numpy()
        action = np.random.choice(self.num_actions, 1, list(action_distribution_np))[0]

        self.hiddens = hiddens

        return action, action_distribution, values

    def update_net(self, net):
        self.net = net

    

    