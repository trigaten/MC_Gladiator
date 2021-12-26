
import gym
import torch
import numpy as np

class OneVersusOneWrapper(gym.Wrapper):
    """Wrapper for a 1v1 version of the multiagent pvp gym.
    This wrapper awards rewards based on agents' health.
    It also converts np observation arrays to pytorch tensors and 
    normalizes its values by dividing by 255
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.a0_health = 20
        self.a1_health = 20

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
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
        reward["agent_0"] = a0_reward
        reward["agent_1"] = a1_reward

        # update agent healths
        self.a0_health = a0_new_health
        self.a1_health = a1_new_health

        # dont need to return health data as obs since we already used it
        # for the reward signal

        # if agent dies lol
        if a0_new_health == 0 or a1_new_health == 0:
            done = True

        # convert to pytorch and normalize
        obs["agent_0"]["pov"] = self.__np_transform(obs["agent_0"]["pov"])
        obs["agent_1"]["pov"] = self.__np_transform(obs["agent_1"]["pov"])
        
        return obs, reward, done, info

    def __np_transform(self, np_array):
        """convert numpy array to pytorch tensor"""
        return torch.from_numpy(np.flip(np_array,axis=0).copy()).permute(2,0,1).unsqueeze(0)/255

    def reset(self):
        obs = self.env.reset()
        obs["agent_0"]["pov"] = self.__np_transform(obs["agent_0"]["pov"])
        obs["agent_1"]["pov"] = self.__np_transform(obs["agent_1"]["pov"])
            
        return obs

class OpponentStepWrapper(gym.Wrapper):
    def __init__(self, env, opponent, actions):
        super().__init__(env)
        self.env = env
        self.opponent = opponent
        self.opponent_obs = None
        self.actions = actions
    
    def step(self, hero_action):
        opponent_action, _, _ = self.opponent(self.opponent_obs)
        dual_action = self.env.action_space.noop()
        op_ac_str = list(self.actions)[opponent_action]
        dual_action["agent_1"][op_ac_str] = self.actions[op_ac_str]
        hero_ac_str = list(self.actions)[hero_action]
        dual_action["agent_0"][hero_ac_str] = self.actions[hero_ac_str]
        obs, reward, done, info = self.env.step(dual_action)
        return obs["agent_0"]["pov"], reward["agent_0"], done, info

    def reset(self):
        obs = self.env.reset()
        self.opponent_obs = obs["agent_1"]["pov"]
        return obs["agent_0"]["pov"]
