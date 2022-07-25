"""MineRL is slow. This fake gym is fast :)"""

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import numpy as np
import gym
import random
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# a bunch of dummy stuff
class ActionSpace():
    def __init__(self) -> None:
        pass
    
    def noop(self):
        return {"agent_0":{}, "agent_1":{}}

class Dummy(gym.Env):
    def __init__(self, *args, **kwargs):
        # self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 255, [64, 64, 3])
        self.action_space = ActionSpace()
    
    def reset(self):
        obs = np.random.randint(0, 255, (64, 64, 3), dtype="uint8")
        ret = {}
        ret["agent_0"] = {"pov":obs, "life_stats":{"life":20}}
        ret["agent_1"] = {"pov":obs, "life_stats":{"life":20}}
        return ret

    def step(self, action):
        obs = np.random.randint(0, 255, (64, 64, 3), dtype="uint8")
        ret = {}
        ret["agent_0"] = {"pov":obs, "life_stats":{"life":19}}
        ret["agent_1"] = {"pov":obs, "life_stats":{"life":15}}
        return ret, {'agent_0': 0.0, 'agent_1': 0.0},random.random() > 0.9,{'agent_0': {}, 'agent_1': {}}

class DummyGym(gym.Env):
    """corresponds to real gym formatting"""
    def __init__(self, *args, **kwargs):
        # self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 255, [3, 64, 64])
        self.action_space = spaces.Discrete(3)

    def reset(self):
        obs = np.random.randint(0, 255, (3, 64, 64), dtype="uint8")

        return obs

    def step(self, action):
        obs = np.random.randint(0, 255, (3, 64, 64), dtype="uint8")

        return obs, 0,random.random() > 0.9, {}

class DummyMAGym(MultiAgentEnv):
    """corresponds to real gym formatting"""
    def __init__(self, action_space, *args, **kwargs):
        # self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 255, [3, 64, 64])
        self.action_space = spaces.Discrete(action_space)
        self._agent_ids = ["agent_0", "agent_1"]
        
    def get_agent_ids(self):
        return self._agent_ids

    def reset(self):
        obs = {}
        obs["agent_0"] = np.random.randint(0, 255, (3, 64, 64), dtype="uint8")
        obs["agent_1"] = np.random.randint(0, 255, (3, 64, 64), dtype="uint8")
        print(obs)
        return obs

    def step(self, action):
        obs = {}
        obs["agent_0"] = np.random.randint(0, 255, (3, 64, 64), dtype="uint8")
        obs["agent_1"] = np.random.randint(0, 255, (3, 64, 64), dtype="uint8")
        rewards = {}
        rewards["agent_0"] = 0
        rewards["agent_1"] = 1
        dones = {}
        dones["agent_0"] = False
        dones["agent_1"] = False
        dones["__all__"] = random.random() > 0.9
        return obs, rewards, dones,{}

if __name__ == "__main__":
    env = DummyGym(7)
    print(env.action_space_sample())