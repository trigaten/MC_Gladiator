
"""This module contains a few wrappers that make observations/rewards from the 1v1 gym useful
and also make the training process easier.
"""

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import gym
import torch
import numpy as np
from gym import spaces
import random
import time
# thanks ray...
from environment.pvpbox_specs import PvpBox


from ray.rllib.env.multi_agent_env import MultiAgentEnv
import threading
threadLock = threading.Lock()
reset_counter = 0

class OneVersusOneWrapper(MultiAgentEnv):
    """Wrapper for a 1v1 version of the multiagent pvp gym.
    This wrapper awards rewards based changes in agents' health.
    It also calculates when an agent dies and terminates the episode.
    It also 'resets' the environment by resetting agents' health, 
    setting their gamemode, and setting their position. This is much more
    efficient than completely resetting the environment.
    """
    def __init__(self, env, actions):
        super().__init__()
        self.env = env
        self.actions = actions
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(0, 255, [3, 64, 64])
        self.START_HEALTH = 40
        self.MAX_STEPS = 600
        self.max_episode_steps = 600
        self.a0_health = self.START_HEALTH
        self.a1_health = self.START_HEALTH
        self.steps = 0

        # count how many times the env has reset
        self.resets = 0
        self.mc_init_commands = [
        "/give @a minecraft:iron_sword 1 0 {Unbreakable:1}", "/gamemode adventure @a"
        ]
        self.mc_reset_commands = [
            "/tp @a 0 5 0", 
            "/effect @a minecraft:instant_health 1 100 true", 
            "/effect @a minecraft:saturation 1 255 true", 
            "/tp MineRLAgent0 0 5 -2",
            "/tp MineRLAgent1 0 5 2 180 0",
        ]
        self._agent_ids = {"agent_0", "agent_1"}
        
    def get_agent_ids(self):
        return self._agent_ids

    def action_space_sample(self):
        return {"agent_0":1,"agent_1":0}

    def step(self, actions):
        dual_action = {"agent_0":{"camera":[0,0]},"agent_1":{"camera":[0,0]}}
        # get the MineRL string action
        hero_action, hero_action_amt = self.actions[actions["agent_0"]]
        # set the action to its value
        dual_action["agent_0"][hero_action] = hero_action_amt
        opponent_action, opponent_action_amt = self.actions[actions["agent_1"]]
        # set the action to its value
        dual_action["agent_1"][opponent_action] = opponent_action_amt
        obs, reward, done, info = self.env.step(dual_action)
        if 'error' in info["agent_0"]:
            print("Error", info)
            print(obs)
            obs = self.env.reboot_env()
        
        a0_new_health = 40
        a1_new_health = 40
        # attempt to deal with issue when life_stats not in the obs
        try:
            # update stored health for both agents
            a0_new_health = obs["agent_0"]["life_stats"]["life"]
            a1_new_health = obs["agent_1"]["life_stats"]["life"]
            
            a0_reward = 0
            a1_reward = 0
            # negative reward upon decrease in agents' health
            # if new health is greater this will be a positive reward (this wont happen as often)
            a0_health_delta = a0_new_health - self.a0_health
            a1_health_delta = a1_new_health - self.a1_health

            # positive reward upon decrease in other agents' health (this would 
            # mean that the agent has damaged the other, which is good) 
            a0_reward += max(-a1_health_delta, 0)
            a1_reward += max(-a0_health_delta, 0)
        except Exception:
            a0_reward = 0
            a1_reward = 0
            # update agent healths
            self.a0_health = self.START_HEALTH
            self.a1_health = self.START_HEALTH

        # set the rewards
        reward = {}
        reward["agent_0"] = a0_reward
        reward["agent_1"] = a1_reward

        

        # dont need to return health data as obs since we already used it
        # for the reward signal

        # if agent dies lol
        dones = {"agent_0":False, "agent_1":False, "__all__":False}
        
        if a0_new_health <= 20 or a1_new_health <= 20 or done or self.steps >= self.MAX_STEPS:
            # extra reward for killing other agent
            # if a0_new_health <= 20:
            #     reward["agent_1"] += 10
            #     reward["agent_0"] -= 10
            # if a1_new_health <= 20:
            #     reward["agent_0"] += 10
            #     reward["agent_1"] -= 10
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
        # reset basic info
        self.steps = 0
        self.a0_health = self.START_HEALTH
        self.a1_health = self.START_HEALTH

        # if the environment has never been reset,
        # we need MineRL to reset/build it
        if self.resets == 0:
            obs = self.env.reset()
            # do init commands
            for mc_command in self.mc_init_commands:
                action = {"agent_0":{"chat":mc_command, "camera":[0,0]},"agent_1":{"camera":[0,0]}}
                obs, reward, done, info = self.env.step(action)

        for mc_command in self.mc_reset_commands:
            obs, reward, done, info = self.env.step({"agent_0":{"chat":mc_command, "camera":[0,0]},"agent_1":{"camera":[0,0]}})
            if done:
                obs = self.env.reset()
                # do init commands
                for mc_command in self.mc_init_commands:
                    action = {"agent_0":{"chat":mc_command, "camera":[0,0]},"agent_1":{"camera":[0,0]}}
                    obs, reward, done, info = self.env.step(action)
                break

        # to pytorch tensors
        new_obs = {}
        new_obs["agent_0"] = self.__np_transform(obs["agent_0"]["pov"])
        new_obs["agent_1"] = self.__np_transform(obs["agent_1"]["pov"])
        
        self.resets+=1
        print("RESETS", self.resets)
        return new_obs

class SuperviserWrapper(gym.Wrapper):
    """
    Adapted from Miffyli's code
    Wrapper that reboots the environment if it happens to crash.

    If environment crashes during step, we return the latest
    observation as a terminal state, and re-init the environment.

    Note: Env should be creatable with gym.make
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.env_id = "PvpBox-v0"

        # Keep track of last observation so we have something
        # valid to give to the agent
        self.last_obs = None

        self.start_time = time.time()

    def reboot_env(self):
        """Re-create the environment from zero"""
        with threadLock:
            global reset_counter
            reset_counter += 1
            print("REBOOT", reset_counter)
        # See if we can close the environment
        try:
            self.env.close()
        except Exception:
            print("Couldnt close env, maybe it is already closed?")
        self.env = PvpBox(agent_count=2).make(instances=[])
        print("MADE NEW ENV")
        try:
            return self.env.reset()
        except Exception as e:
            print("THATS A RESET FAIL", e)

    def reset(self, **kwargs):
        try:
            obs = self.env.reset(**kwargs)
        except Exception as e:
            print("[{}, Superviser] Environment crashed with '{}'".format(
                time.time() - self.start_time, str(e))
            )

            # Re-create the environment
            obs = self.reboot_env()
        self.last_obs = obs
        return obs

    def step(self, action):
        obs, reward, done, info = None, None, None, None

        try:
            obs, reward, done, info = self.env.step(action)
            self.last_obs = obs
        except Exception as e:
            print("[{}, Superviser] Environment crashed with '{}'".format(
                time.time() - self.start_time, str(e))
            )
            # Create something to return
            obs = self.last_obs
            reward = 0
            done = True
            info = {}
            # Re-create the environment
            self.reboot_env()
        if info and "agent_0" in info and info["agent_0"]:
            print("INFO", info)
        return obs, reward, done, info
