from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from typing import List
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
import coloredlogs
import logging
coloredlogs.install(logging.DEBUG)
from minerl.human_play_interface.human_play_interface import HumanPlayInterface

from src.MCGladiator.flat_env import HumanSurvivalMultiplayer

env = HumanSurvivalMultiplayer("127.0.0.1:25565", "human").make()

env = HumanPlayInterface(env)
print(env.reset())

done = False

while not done:
    obs, reward, done, _ = env.step()
    print(obs["life_stats"])