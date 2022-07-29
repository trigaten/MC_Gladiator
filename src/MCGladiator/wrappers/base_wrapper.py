from typing import Dict

import gym
from gym import spaces

class BaseWrapper(gym.Wrapper):
    """Base wrapper for a 1v1 pvp env.

    * Calculate rewards based on changes in agents' health.

    * Calculate when an agent dies and terminate the episode.

    :param env: The environment to wrap.
    :param bool simple_actions: Whether to use simple actions or not.
    """
    def __init__(self, env, simple_actions):
        super().__init__(env)
        self.env = env
        self.simple_actions = simple_actions
        self.steps = 0
        self.agent_healths = self.get_default_healths()
        if simple_actions:
            self.actions = [("attack", 1), ("left", 1), ("back", 1), ("right", 1), ("forward", 1), ("camera", [0,15]), ("camera", [0,-15])]

        # commands to be executed at the reset of the env
        self.reset_commands = [
            "/give @a minecraft:iron_sword 1 0 {Unbreakable:1}",
            "/tp MineRLAgent0 -2 5 0 270 0",
            "/tp MineRLAgent1 2 5 0 90 0",
            "/difficulty hard"
        ]

        self._agent_ids = {"agent_0", "agent_1"}
        
    def get_noop(self):
        """Return an action which does nothing, but sets enough information
        to allow for MineRL `render()`ing"""
        return {"agent_0":{"camera":[0,0]},"agent_1":{"camera":[0,0]}}

    def compute_rewards(self, a0_new_health:float, a1_new_health:float):
        """Compute reward based on health delta.
        
        Agents are inversely rewarded for the opponents' change in health.
        """

        reward = {}
        reward["agent_0"] = self.agent_healths["agent_1"] - a1_new_health
        reward["agent_1"] = self.agent_healths["agent_0"] - a0_new_health

        # normalize rewards by iron sword critical hit damage
        critical_hit_damage = 1.5 * 6
        reward["agent_0"] /= critical_hit_damage
        reward["agent_1"] /= critical_hit_damage

        return reward

    def compute_env_dones(self, done:bool, a0_new_health:float, a1_new_health:float):
        dones = {"agent_0":False, "agent_1":False, "__all__":False}
        
        # TODO: this should be the only case of doneness    
        # check if an agent has died
        if done or a0_new_health <= 0 or a1_new_health <= 0:
            dones = {"agent_0":True, "agent_1":True, "__all__":True}
        return dones

    """Take a dict of numerical `actions` (e.g. `{"agent_0:2", "agent_1:5"}`) and
    steps the environment with the corresponding MineRL actions"""
    def step(self, actions:Dict[str, int]):
        # 1. start with a noop action for 2 agents
        dual_action = self.get_noop()

        # 2. put proper MineRL action names into dual_action
        
        # get the MineRL string actions
        a0_action_name, a0_action_amt = self.actions[actions["agent_0"]]
        a1_action_name, a1_action_amt = self.actions[actions["agent_1"]]

        # set the actions to correct values
        dual_action["agent_0"][a0_action_name] = a0_action_amt
        dual_action["agent_1"][a1_action_name] = a1_action_amt

        # 3. step the env
        obs, _, done, info = self.env.step(dual_action)

        # if info:
        #     print(info)
        
        # 4. compute rewards
        a0_new_health = obs["agent_0"]["life_stats"]["life"]
        a1_new_health = obs["agent_1"]["life_stats"]["life"]
        print(obs["agent_0"]["life_stats"]["life"])
        rewards = self.compute_rewards(a0_new_health = a0_new_health, a1_new_health = a1_new_health)
        
        # 5. check for doneness of environment
        dones = self.compute_env_dones(done, a0_new_health, a1_new_health)

        # 6. set obs
        pov_obs = {}
        pov_obs["agent_0"] = obs["agent_0"]["pov"]
        pov_obs["agent_1"] = obs["agent_1"]["pov"]

        # 7. update variables for next step
        self.agent_healths = {"agent_0":a0_new_health, "agent_1":a1_new_health}
        self.steps+= 1
        
        return pov_obs, rewards, dones, info

    def reset(self):
        # 1. reset basic info
        self.steps = 0
        self.agent_healths = self.get_default_healths()

        # 2. reset env
        obs = self.env.reset()
        
        # 3. start agents in correct positions, with swords
        for mc_command in self.reset_commands:
            # put action into dict space formatting
            action = self.get_noop()
            action["agent_0"]["chat"] = mc_command
            # execute action
            obs, _, done, info = self.env.step(action)

        # 4. get the last obs
        new_obs = {}
        new_obs["agent_0"] = obs["agent_0"]["pov"]
        new_obs["agent_1"] = obs["agent_1"]["pov"]
        
        return new_obs

    def get_default_healths(self):
        default_health = self.get_default_health()
        return {"agent_0":default_health, "agent_1":default_health}

    def get_default_health(self):
        return 40

    def get_agent_ids(self):
        return self._agent_ids

    def get_agent_healths(self):
        return self.agent_healths
    