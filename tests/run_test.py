# from typing import Dict
from collections import OrderedDict
from minerl.herobraine.hero.spaces import Dict, Discrete, Box, Text
from MCGladiator.envs import PvpBox

class TestEnvironment():
    env = PvpBox(agent_count=2).make(instances=[])
    
    _ = env.reset()
    # check agent count
    assert len(env.instances) == 2
    # check step works
    env.step({"agent_0":{}, "agent_1":{}})

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



