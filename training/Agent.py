__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import torch
import numpy as np

class Agent():
    """Class which takes observations and returns the selected action, 
    the prob dist over possible actions, and the state values"""
    def __init__(self, net, grad=True):
        """
        :param grad: whether or not to perform operation with gradients.
        The opponent agent shouldnt use gradients, but the training agent should.
        """
        self.net = net
        self.grad = grad
        # need to store last hidden layer since the model uses a GRU
        self.hiddens = None
        self.num_actions = net.num_outputs
    
    def __call__(self, states):
        # perform forward pass through network
        if self.grad:
            action_distribution, values, hiddens = self.net(states, self.hiddens)
        else:
            with torch.no_grad():
                action_distribution, values, hiddens = self.net(states, self.hiddens)

        # convert the action distribution to numpy
        action_distribution_np = torch.reshape(action_distribution, (self.num_actions,)).detach().cpu().numpy()
        # sample according to the probability distribution
        action = np.random.choice(self.num_actions, 1, list(action_distribution_np))[0]

        # update hiddens
        self.hiddens = hiddens

        return action, action_distribution, values

    def update_net(self, net):
        self.net = net

    

    