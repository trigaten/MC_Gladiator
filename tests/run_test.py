from minerl.herobraine.hero.spaces import Dict, Discrete, Box, Text

from MCGladiator.envs import PvpBox
from MCGladiator.wrappers.base_wrapper import BaseWrapper

class TestBaseEnvironment():
    def test_base_box_env(self):
        env = PvpBox(agent_count=2).make(instances=[])
        
        obs = env.reset()
        # check agent count
        assert len(env.instances) == 2
        # check life stats in obs
        assert "life_stats" in obs["agent_0"]
        assert "life_stats" in obs["agent_1"]

        # check step works
        env.step({"agent_0":{}, "agent_1":{}})

class TestBaseWrapper():
    def __init__(self) -> None:
        env = PvpBox(agent_count=2).make(instances=[])
        self.wrapped_env = BaseWrapper(env=env, simple_actions=True)
        self.obs = env.reset()

    def test_done_computation(self):
        # test done computation
        dones_false = {"agent_0":False, "agent_1":False, "__all__":False}
        dones_true = {"agent_0":True, "agent_1":True, "__all__":True}
        
        assert (self.wrapped_env.compute_env_dones(done=False, a0_new_health=20, a1_new_health=20) == dones_false)
        assert self.wrapped_env.compute_env_dones(done=True, a0_new_health=20, a1_new_health=20) == dones_true
        assert self.wrapped_env.compute_env_dones(done=True, a0_new_health=0, a1_new_health=0) == dones_true
        assert self.wrapped_env.compute_env_dones(done=False, a0_new_health=0, a1_new_health=20) == dones_true
        assert self.wrapped_env.compute_env_dones(done=False, a0_new_health=20, a1_new_health=0) == dones_true
        assert self.wrapped_env.compute_env_dones(done=False, a0_new_health=0, a1_new_health=0) == dones_true

    def test_reward_computation(self):
        # test reward computation
        # agent_1 should get reward, while agent_0 should get none
        rewards = self.wrapped_env.compute_rewards(a0_new_health=15, a1_new_health=20)
        assert rewards["agent_1"] > 0 and rewards["agent_0"] == 0

        # agent_0 should get reward, while agent_1 should get none
        rewards = self.wrapped_env.compute_rewards(a0_new_health=20, a1_new_health=15)
        assert rewards["agent_1"] == 0 and rewards["agent_0"] > 0

        # both agents should get negative reward
        rewards = self.wrapped_env.compute_rewards(a0_new_health=21, a1_new_health=21)
        assert rewards["agent_0"] < 0 and rewards["agent_1"] < 0

        # test normalization
        rewards = self.wrapped_env.compute_rewards(a0_new_health=11, a1_new_health=20)
        assert rewards["agent_1"] == 1

        # test noop stepping
        # noop = self.wrapped_env.get_noop()
        self.wrapped_env.step({"agent_0":2, "agent_1":0})

    
    

    # action_space = env.action_space
    # OrderedDict([

    #     ("agent_0",
    #         OrderedDict([
    #             ("attack", Discrete(2)), ("back", Discrete(2)),
    #             ("camera", Box(low=-180.0, high=180.0, shape=(2,))),
    #             ("chat", Text(1,)), ("forward", Discrete(2)), ("jump", Discrete(2)),
    #             ("left", Discrete(2)), ("right", Discrete(2)), ("sneak", Discrete(2)),
    #             ("sprint", Discrete(2))
    #         ])
    #     ),

    # ])



