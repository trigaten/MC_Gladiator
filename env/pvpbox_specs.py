# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS
from minerl.herobraine.hero.handler import Handler
from typing import List

import minerl.herobraine
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.env_spec import EnvSpec

PVPBOX_DOC = """
In pvp box, two agents fight in a boxed in area
"""

PVPBOX_LENGTH = 8000

class PvpBox(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'PvpBox-v0'

        super().__init__(*args,
                         max_episode_steps=PVPBOX_LENGTH, reward_threshold=64.0,
                         **kwargs)

    def create_rewardables(self) -> List[Handler]:
        return []

    # make agent spawn with iron sword
    def create_agent_start(self) -> List[Handler]:
        return [
            handlers.SimpleInventoryAgentStart([
                dict(type="iron_sword", quantity=1)
            ])
        ]

    # make it so agent receives life stats data in addition
    # to the image data it receives at each time step
    # life stats data will be used for calculating +/- 
    # reward agent gets from doing/taking damage
    def create_agent_handlers(self) -> List[Handler]:
        return [handlers.ObservationFromLifeStats()]

    # make the agent spawn on a super flat world
    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.FlatWorldGenerator(generatorString="2;7,2x3,2;1;")
            # handlers.DrawingDecorator('<DrawSphere x="-50" y="20" z="0" radius="10" type="obsidian"/>')
                        # 95 227 186

            # handlers.DrawingDecorator('<DrawSphere x="0" y="10" z="0" radius="10" type="obsidian"/>'),
            # handlers.DrawingDecorator('<DrawSphere x="-50" y="10" z="0" radius="10" type="obsidian"/>'),
            # handlers.DrawingDecorator('<DrawSphere x="-50" y="10" z="-50" radius="10" type="obsidian"/>'),
            # handlers.DrawingDecorator('<DrawSphere x="50" y="10" z="-50" radius="10" type="obsidian"/>'),
            # handlers.DrawingDecorator('<DrawSphere x="50" y="10" z="50" radius="10" type="obsidian"/>')

        ]

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(
                (PVPBOX_LENGTH * MS_PER_STEP)),
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
        return folder == 'survivalpvpbox'

    def get_docstring(self):
        return PVPBOX_DOC
