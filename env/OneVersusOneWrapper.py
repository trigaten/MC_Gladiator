"""Wrapper for a 1v1 version of the multiagent pvp gym.
This wrapper awards rewards based on agents' health
It also divides pov pixels by 255
"""
import gym

class OneVersusOneWrapper(gym.Wrapper):
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
        # also divide pov pixels by 255
        obs["agent_0"] = a0_new_health = obs["agent_0"]["pov"]/255
        obs["agent_1"] = a0_new_health = obs["agent_1"]["pov"]/255

        return obs, reward, done, info
