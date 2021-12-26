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

agent_actions = {"attack":1, "left":1, "right":1}
num_actions = len(agent_actions)
env = OneVersusOneWrapper(Dummy())
opponent = Agent(Discrete_PPO_net(num_actions), False)
env = OpponentStepWrapper(env, opponent, agent_actions)

hero = Agent(Discrete_PPO_net(num_actions), True)

optimizer = optim.Adam(hero.net.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss()


BATCH_SIZE = 8
EPOCHS = 100

# training loop
while True:
    value_batch = []
    action_batch = []
    action_distribution_batch = []
    reward_batch = []
    # obtain batch
    for i in range(BATCH_SIZE):
        obs = env.reset()
        values = []
        actions_taken = []
        action_distributions = []
        rewards = []
        states = []
        steps = 0

        done = False
        while not done:
            steps += 1
            # get hero action
            action, action_distribution, value = hero(obs)
            # append items
            states.append(obs)
            actions_taken.append(action)
            action_distributions.append(action_distribution)
            values.append(value)

            # step with action
            obs, reward, done, info = env.step(action)

        value_batch.append(values)
        action_batch.append(actions_taken)
        action_distribution_batch.append(action_distributions)
        reward_batch.append(rewards)

    # train on batch

    

    


