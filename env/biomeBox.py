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
    
    def create_rewardables(self) -> List[Handler]:
        return []

    # make agent spawn with iron sword
    def create_agent_start(self) -> List[Handler]:
        return [
            handlers.SimpleInventoryAgentStart([
                dict(type="iron_sword", quantity=1)
            ]),
             handlers.AgentStartPlacement(0, 5, 0, 0, 0)
        ]
   
    def create_agent_handlers(self) -> List[Handler]:
        return []

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

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(
                (BIOMEBOX_LENGTH * MS_PER_STEP)),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=True
            )
        ]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'biomeBox' #change 

    def get_docstring(self):
        return BIOMEBOX_DOC
