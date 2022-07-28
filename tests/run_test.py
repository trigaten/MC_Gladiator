from minerl.herobraine.hero.spaces import Dict, Discrete, Box, Text

from MCGladiator.envs import PvpBox
from MCGladiator.wrappers.base_wrapper import BaseWrapper

class TestEnvironments():
    def test_base_box_env(self):
        env = PvpBox(agent_count=2).make(instances=[])
        
        obs = env.reset()
        # check agent count
        assert len(env.instances) == 2
        # check life stats in obs
        assert "life stats" in obs["agent_0"]
        assert "life_stats" in obs["agent_1"]

        noop = env.get_noop()
        # check step works
        env.step(noop)

    def test_base_wrapper(self):
        env = PvpBox(agent_count=2).make(instances=[])
        wrapped_env = BaseWrapper(env=env, simple_actions=True)

        # test done computation
        dones_false = {"agent_0":False, "agent_1":False, "__all__":False}
        dones_true = {"agent_0":True, "agent_1":True, "__all__":True}
        
        assert wrapped_env.compute_env_dones(done=False, a0_new_health=20, a1_new_health=20) == dones_false
        assert wrapped_env.compute_env_dones(done=True, a0_new_health=20, a1_new_health=20) == dones_true
        assert wrapped_env.compute_env_dones(done=True, a0_new_health=0, a1_new_health=0) == dones_true
        assert wrapped_env.compute_env_dones(done=False, a0_new_health=0, a1_new_health=20) == dones_true
        assert wrapped_env.compute_env_dones(done=False, a0_new_health=20, a1_new_health=0) == dones_true
        assert wrapped_env.compute_env_dones(done=False, a0_new_health=0, a1_new_health=0) == dones_true
    # def test_wrapped

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



