from MCGladiator.server import Server
import subprocess
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from typing import List
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
import coloredlogs
import logging
import time
# coloredlogs.install(logging.DEBUG)
from minerl.human_play_interface.human_play_interface import HumanPlayInterface
import subprocess
from src.MCGladiator.flat_env import HumanSurvivalMultiplayer

# spin up minecraft server


server = Server("mc")
human_env = HumanSurvivalMultiplayer("127.0.0.1:25565", "human").make()
human_env = HumanPlayInterface(human_env)

bot_env = HumanSurvivalMultiplayer("127.0.0.1:25565", "ai").make()

human_env.reset()
bot_env.reset()

server.execute("tp human -2 4 0 270 0")
server.execute("tp ai 2 4 0 90 0")
server.execute("gamerule doNaturalRegen false")
server.execute("give @a minecraft:iron_sword 1")
time.sleep(2)
server.execute("fill -3 3 -3 3 7 3 gold_block outline")
server.execute("fill -3 7 -3 3 7 3 air outline")
server.execute("effect give @a instant_health 1 50")
done = False

while not done:
    human_obs, reward, done, _ = human_env.step()
    obs, reward, done, _ = bot_env.step({"camera":[1,1], "jump":1,"forward":1})

    if human_obs["life_stats"]['life'] <= 10:
        server.execute("/tp human -2 4 0 270 0")
        server.execute("/tp ai 2 4 0 90 0")

    print(human_obs["life_stats"])