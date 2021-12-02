""" BIOME BOX FOR GLADIATOR PROJECT | Created based on https://github.com/trigaten/minerl/blob/dev/docs/source/tutorials/custom_environments.rst """

from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS
from minerl.herobraine.hero.handler import Handler
from typing import List

import minerl.herobraine.hero.handlers as handlers

__author__ = "Adam Yang"
__email__ = "adamfyang@gmail.com"

BIOMEBOX_DOC = """
Creates enclosed region within natural MC landscape
"""

BIOMEBOX_LENGTH = 8000

class biomeBox(SimpleEmbodimentEnvSpec):
    def __init__(self, name, *args, resolution=..., **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'biomeBox-v0'

        super().__init__(*args,
                        max_episode_steps=BIOMEBOX_LENGTH, reward_threshold=100.0,
                        **kwargs)

    #Creates a default world with no preset gerenation (random world) 
    # with a 5x5 open space enclosed by bedrock stacked from y=0 to y=50.
    #Adapted from survival_specs (minerl\herobraine\env_specs\survival_specs.py)
    def create_server_world_generators(self) -> List[Handler]:

        return [
            handlers.DefaultWorldGenerator(force_reset="true", generator_options=""),
            handlers.DrawingDecorator("""
            <DrawCuboid x1="3" y1="0" z1="3" x2="3" y2="50" z2="-3" type="bedrock"/>
            <DrawCuboid x1="3" y1="0" z1="3" x2="-3" y2="50" z2="3" type="bedrock"/>
            <DrawCuboid x1="-3" y1="0" z1="-3" x2="3" y2="50" z2="-3" type="bedrock"/>
            <DrawCuboid x1="-3" y1="0" z1="-3" x2="-3" y2="50" z2="3" type="bedrock"/>
        """)
        ]

