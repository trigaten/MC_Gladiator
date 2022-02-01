""" NATURAL LANDSCAPE FOR GLADIATOR PROJECT | Created based on https://github.com/trigaten/minerl/blob/dev/docs/source/tutorials/custom_environments.rst """

from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS
from minerl.herobraine.hero.handler import Handler
from typing import List

import minerl.herobraine.hero.handlers as handlers

NATURAL_DOC = """
Creates region within natural MC landscape
"""

NATURAL_LENGTH = 8000

class naturalArena(SimpleEmbodimentEnvSpec):
    def __init__(self, name, *args, resolution=..., **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'naturalArena-v0'

        super().__init__(*args,
                        max_episode_steps=NATURAL_LENGTH, reward_threshold=100.0,
                        **kwargs)


    def create_rewardables(self) -> List[Handler]:
        return []


    def create_agent_start(self) -> List[Handler]:
        """Make agents have iron swords and start in a close area."""
        return [
            #ADD STARTING AGENT INVENTORY 
            handlers.StartingHealthAgentStart(max_health=40, health=40),
            handlers.AgentStartNear("MineRLAgent0", min_distance=2, max_distance=20, max_vert_distance=10)
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return []

    # make it so agent receives life stats data in addition
    # to the image data it receives at each time step
    # life stats data will be used for calculating +/- 
    # reward agent gets from doing/taking damage
    def create_observables(self) -> List[Handler]:
        return super().create_observables() + [
            handlers.ObservationFromLifeStats()
        ]

    # Creates a default world with no preset gerenation (random world) 
    def create_server_world_generators(self) -> List[Handler]:

        return [
            handlers.DefaultWorldGenerator(force_reset="true", generator_options=""),
        ]
        
    def create_actionables(self) -> List[Handler]:
        """Will be used to reset agents health, etc. without resetting the entire environment"""
        return super().create_actionables() + [
            handlers.ChatAction()
        ]

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(
                (NATURAL_LENGTH * MS_PER_STEP)),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]
     
    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        """Stop time"""
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False,
                start_time=0
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=False
            ),
            handlers.AgentStartPlacement(0, 60, 0, 0, 0)


        ]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'environment'

    def get_docstring(self):
        return NATURAL_DOC

class PvpBoxNoQuit(naturalArena):
    """env doesnt reset when another agent dies"""
    def create_server_quit_producers(self):
        return [

        ]


