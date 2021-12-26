from minerl.env.malmo import InstanceManager
import sys
sys.path.append("..")

from env.pvpbox_specs import PvpBox
from env.wrappers import OneVersusOneWrapper
from env.wrappers import OpponentStepWrapper

from model import Discrete_PPO_net
from Agent import Agent

import torch
import torch.optim as optim
import numpy as np

class MultiPvpBox(PvpBox):
    # This version of treechop doesn't terminate the episode 
    # if the other agent quits/dies (or gets the max reward)
    # def create_server_quit_producers(self):
    #     return []
    pass

agent_actions = {"attack":1, "left":1, "right":1}
num_actions = len(agent_actions)
env_spec = MultiPvpBox(agent_count=2)

# IF you want to use existing instances use this!
# instances = [
#     InstanceManager.add_existing_instance(9001),
#     InstanceManager.add_existing_instance(9002)]
instances = []

env = OneVersusOneWrapper(env_spec.make(instances=instances))
opponent = Agent(Discrete_PPO_net(num_actions), False)
env = OpponentStepWrapper(env, opponent, agent_actions)

hero = Agent(Discrete_PPO_net(num_actions), False)

optimizer = optim.Adam(hero.net.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss()

value_batch = []
action_batch = []
# iterate desired episodes
while True:
    obs = env.reset()
    values = []
    actions = []
    rewards = []
    steps = 0

    done = False
    while not done:
        steps += 1
        action, action_distribution, values = hero(obs)
        # env.render()
        # actions = env.env.action_space.no_op()
        # for agent in actions:
        #     actions[agent]["forward"] = 1
        #     actions[agent]["attack"] = 1
        #     actions[agent]["camera"] = [0, 0.1]

        obs, reward, done, info = env.step(action)


