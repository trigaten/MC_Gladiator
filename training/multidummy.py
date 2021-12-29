"""Contains the training loop/routine. This is the current development version, which is
relying on the Dummy environment, since interaction with the actual environment is working, but
takes a while. """

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import sys
sys.path.append("..")

from env.dummy_spec import Dummy
from env.wrappers import OneVersusOneWrapper
from env.wrappers import OpponentStepWrapper

from ray.rllib.agents.ppo import PPOTrainer

from model import Discrete_PPO_net
from Agent import Agent

import torch
import torch.optim as optim

from utils import GAE_adv

# similar to PPO paper
LAMBDA = 0.95
DISCOUNT = 0.99
EPSILON = 0.2
BATCH_SIZE = 8
EPOCHS = 10
ENTROPY = 0.01
LR = 3e-4

agent_actions = {"attack":1, "left":1, "right":1}
num_actions = len(agent_actions)
env = OneVersusOneWrapper(Dummy())
opponent = Agent(Discrete_PPO_net(num_actions), False)
env = OpponentStepWrapper(env, opponent, agent_actions)

hero = Agent(Discrete_PPO_net(num_actions), True)

optimizer = optim.Adam(hero.net.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

# training loop

# policy = ts.policy.PPOPolicy(actor, critic, optimizer, distribution, value_clip=False, advantage_normalization=False)