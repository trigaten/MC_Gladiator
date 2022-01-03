
"""This module contains a few wrappers that make observations/rewards from the 1v1 gym useful
and also make the training process easier.
"""

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import gym
import torch
import numpy as np
from gym import spaces

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class OneVersusOneWrapper(MultiAgentEnv):
    """Wrapper for a 1v1 version of the multiagent pvp gym.
    This wrapper awards rewards based changes in agents' health.
    It also converts np observation arrays to pytorch tensors and 
    normalizes its values by dividing by 255.
    It also calculates when an agent dies and terminates the episode.
    """
    def __init__(self, env, actions):
        super().__init__()
        self.env = env
        self.actions = actions
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 255, [3, 64, 64])
        self.a0_health = 20
        self.a1_health = 20
        self.steps = 0

    def step(self, actions):
        dual_action = self.env.action_space.noop()
        # get the MineRL string action
        hero_action = list(self.actions)[actions["agent_0"]]
        # set the action to its value
        dual_action["agent_0"][hero_action] = self.actions[hero_action]
        opponent_action = list(self.actions)[actions["agent_1"]]
        # set the action to its value
        dual_action["agent_1"][opponent_action] = self.actions[opponent_action]
        obs, reward, done, info = self.env.step(dual_action)
        # update stored health for both agents
        a0_new_health = obs["agent_0"]["life_stats"]["life"]
        a1_new_health = obs["agent_1"]["life_stats"]["life"]

        # negative reward upon decrease in agents' health
        # if new health is greater this will be a positive reward (this wont happen as often)
        a0_reward = a0_new_health - self.a0_health
        a1_reward = a1_new_health - self.a1_health

        # positive reward upon decrease in other agents' health (this would 
        # mean that the agent has damaged the other, which is good) 
        a0_reward -= a1_reward
        a1_reward -= a0_reward

        # set the rewards
        reward = {}
        reward["agent_0"] = a0_reward
        reward["agent_1"] = a1_reward

        # update agent healths
        self.a0_health = a0_new_health
        self.a1_health = a1_new_health

        # dont need to return health data as obs since we already used it
        # for the reward signal

        # if agent dies lol
        dones = {"agent_0":False, "agent_1":False, "__all__":False}
        
        if a0_new_health == 0 or a1_new_health == 0 or done or self.steps > 10:
            dones = {"agent_0":True, "agent_1":True, "__all__":True}
        # convert to pytorch and normalize
        new_obs = {}
        new_obs["agent_0"] = self.__np_transform(obs["agent_0"]["pov"])
        new_obs["agent_1"] = self.__np_transform(obs["agent_1"]["pov"])

        self.steps+= 1
        return new_obs, reward, dones, {}

    def __np_transform(self, np_array):
        """convert numpy array to pytorch tensor and normalize it"""
        # return torch.from_numpy(np.flip(np_array,axis=0).copy()).permute(2,0,1).unsqueeze(0)/255
        return np.flip(np_array,axis=0).transpose(2,0,1)

    def reset(self):
        print("RESETTING")
        obs = self.env.reset()
        print(obs)
        # to pytorch tensors
        new_obs = {}
        new_obs["agent_0"] = self.__np_transform(obs["agent_0"]["pov"])
        new_obs["agent_1"] = self.__np_transform(obs["agent_1"]["pov"])
        print("NEW", new_obs)
        return new_obs

class OpponentStepWrapper(gym.Wrapper):
    """This wrapper makes the environment look like a single agent environment.
    The action/observation space of the opponent agent does not get returned.
    This wrapper also ensures that the opponent agent acts according to its own
    neural network.
    """
    def __init__(self, env, opponent, actions):
        """
        :param actions: a dictionary of strings to values corresponding to 
        MineRL actions (and how much of the action to perform)
        """
        super().__init__(env)
        self.env = env
        self.opponent = opponent
        self.opponent_obs = None
        self.actions = actions
    
    def step(self, hero_action):
        """
        :param hero_action: an integer value corresponding to an index in
        self.actions
        """
        # get the action, we dont care about other information
        # because we arent training this agent
        opponent_action, _, _ = self.opponent(self.opponent_obs)
        # get the default noop action
        dual_action = self.env.action_space.noop()
        # get the MineRL string action
        op_ac_str = list(self.actions)[opponent_action]
        # set the action to its value
        dual_action["agent_1"][op_ac_str] = self.actions[op_ac_str]
        # get the MineRL string action for the hero action
        hero_ac_str = list(self.actions)[hero_action]
        # set the action to its value
        dual_action["agent_0"][hero_ac_str] = self.actions[hero_ac_str]
        # perform a step
        obs, reward, done, info = self.env.step(dual_action)
        # return information for agent_0
        return obs["agent_0"]["pov"], reward["agent_0"], done, info

    def reset(self):
        """just returns observation from first agent"""
        obs = self.env.reset()
        self.opponent_obs = obs["agent_1"]["pov"]
        return obs["agent_0"]["pov"]
