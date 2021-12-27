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
while True:
    value_batch = []
    action_batch = []
    action_distribution_batch = []
    adv_batch = []
    # obtain batch
    for i in range(BATCH_SIZE):
        obs = env.reset()
        values = []
        actions_taken = []
        action_distributions = []
        rewards = []
        steps = 0

        done = False
        while not done:
            steps += 1
            # get hero action
            action, action_distribution, value = hero(obs)
            # append items
            actions_taken.append(action)
            action_distributions.append(action_distribution)
            values.append(value)

            # step with action
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            # print(reward)

        value_batch.append(values)
        action_batch.append(actions_taken)
        action_distribution_batch.append(action_distributions)
        adv_batch.append(GAE_adv(rewards, values, DISCOUNT, LAMBDA))

    # train on batch
    for i in range(EPOCHS):
        action_batch = torch.FloatTensor(action_batch)
        exit()


    

    


