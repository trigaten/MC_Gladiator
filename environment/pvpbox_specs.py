"""
Adapted from https://github.com/minerllabs/minerl/blob/dev/minerl/herobraine/env_specs/treechop_specs.py
"""

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS
from minerl.herobraine.hero.handler import Handler
from typing import List

import minerl.herobraine.hero.handlers as handlers

PVPBOX_DOC = """
In pvp box, two agents fight in a boxed-in area
"""

PVPBOX_LENGTH = 80000

class PvpBox(SimpleEmbodimentEnvSpec):

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'PvpBox-v0'

        super().__init__(*args,
                         max_episode_steps=PVPBOX_LENGTH, reward_threshold=64.0,
                         **kwargs)
    
    def create_rewardables(self) -> List[Handler]:
        """There is no handler which provides positive/negative reward
        based on damage taken/dealt. This functionality will be implemented as
        a wrapper.
        """
        return []

    def create_agent_start(self) -> List[Handler]:
        """Make agents have iron swords and start at the specified location."""
        return [
            # handlers.SimpleInventoryAgentStart([
            #     dict(type="iron_sword", quantity=1)
            # ]),
            handlers.AgentStartPlacement(0, 5, 0, 0, 0),
            handlers.StartingHealthAgentStart(max_health=40, health=40),

            # dont ask...
            # handlers.SimpleInventoryAgentStart([
            #     {'type':'iron_boots', 'quantity':1} for i in range(140)
            # ]),
        ]

    def create_actionables(self) -> List[Handler]:
        """Will be used to reset agents health, etc. without resetting the entire environment"""
        return super().create_actionables() + [
            handlers.ChatAction()
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return []
    
    def create_observables(self) -> List[Handler]:
        """
        Make it so agent receives life stats data in addition
        to the image data it receives at each time step
        life stats data will be used for calculating +/- 
        reward agent gets from doing/taking damage
        """
        return super().create_observables() + [
            handlers.ObservationFromLifeStats()
        ]

    def create_server_world_generators(self) -> List[Handler]:
        """Make the agent spawn on a super flat world
        Also draw a box around it"""
        return [
            # make it so world doesnt reset (this wastes compute)
            # , force_reset="false"
            handlers.FlatWorldGenerator(generatorString="2;7,2x3,2;1;"),
            handlers.DrawingDecorator("""<DrawCuboid x1="3" y1="4" z1="3" x2="3" y2="6" z2="-3" type="gold_block"/>
            <DrawCuboid x1="3" y1="4" z1="3" x2="-3" y2="6" z2="3" type="gold_block"/>
            <DrawCuboid x1="-3" y1="4" z1="-3" x2="3" y2="6" z2="-3" type="gold_block"/>
            <DrawCuboid x1="-3" y1="4" z1="-3" x2="-3" y2="6" z2="3" type="gold_block"/>"""),
          ]

    def create_server_quit_producers(self) -> List[Handler]:
        """Quit on server time up or when any agent finishes"""
        return [
            handlers.ServerQuitFromTimeUp(
                (PVPBOX_LENGTH * MS_PER_STEP)),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        """time will not pass"""
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=False
            )
        ]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'survivalpvpbox'

    def get_docstring(self):
        return PVPBOX_DOC

class PvpBoxNoQuit(PvpBox):
    """env doesnt reset when another agent dies"""
    def create_server_quit_producers(self):
        return [

        ]