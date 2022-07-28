from minerl.herobraine.hero.spaces import Dict, Discrete, Box, Text

from MCGladiator.envs import PvpBox
from MCGladiator.wrappers import BaseWrapper

class TestEnvironments():
    def test_base_box_env():
        env = PvpBox(agent_count=2).make(instances=[])
        
        _ = env.reset()
        # check agent count
        assert len(env.instances) == 2
        # check life stats in obs spaces
        assert "life_stats" in env.observation_space.spaces["agent_0"]
        assert "life_stats" in env.observation_space.spaces["agent_1"]

        # check step works
        env.step({"agent_0":{}, "agent_1":{}})

    def test_base_wrapper():
        pass
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



