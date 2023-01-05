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
import os
import pickle
import random

server = Server("mc")
human_env = HumanSurvivalMultiplayer("127.0.0.1:25565", "human").make()
human_env = HumanPlayInterface(human_env)

bot_env = HumanSurvivalMultiplayer("127.0.0.1:25565", "ai").make()

human_env.reset()
bot_env.reset()
import json

def reset():
    server.execute("clear @a")
    server.execute("tp human -2.5 5 0.5 270 0")
    server.execute("tp ai 3.5 5 0.5 90 0")
    server.execute("gamerule naturalRegeneration false")
    server.execute("give @a minecraft:iron_sword 1")
    server.execute("effect give @a instant_health 1 200")
    server.execute("effect give @a saturation 1 255")
    time.sleep(0.1)

done = False
reset()
server.execute("fill -4 3 -4 4 7 4 gold_block outline")
server.execute("fill -4 7 -4 4 7 4 air outline")

for i in range(100):
    if not os.path.exists("episodes/" + str(i)):
        os.mkdir("episodes/" + str(i))
                
    with open("episodes/" + str(i) + '/human_obs', 'ab') as human_obs_file: 
        with open("episodes/" + str(i) + "/ai_obs", 'ab') as ai_obs_file:        
            with open("episodes/" + str(i) + "/stats.json", "a") as stats:
                stats.write("[")
                reset()
                print("RESET")
                done = False
                step = 0
                while not done:
                    human_obs, h_reward, h_done, human_info = human_env.step()
                    b_obs, b_reward, b_done, _ = bot_env.step({"camera":[10*random.random()-5,10*random.random()-5], "jump": 1 if random.random()>0.5 else 0,"forward":1})

                    if human_obs["life_stats"]['life'] <= 10 or b_obs["life_stats"]["life"] <= 10:
                        done = True
                    # print("----")
                    # print(type(human_obs["life_stats"]['life']))
                    # print(b_obs["life_stats"]['life'])
                    # print(b_obs["pov"])
                    print(b_obs["location_stats"])
                    pickle.dump(b_obs["pov"], ai_obs_file)
                    pickle.dump(human_obs["pov"], human_obs_file)
                    if not done:
                        ap = ","
                    else:
                        ap = ""
                    stats.write(json.dumps({step:{"human": {"life": human_obs["life_stats"]['life'].item(), "action":human_info["taken_action"]}, "ai": {"life": b_obs["life_stats"]['life'].item()}}}) + ap)
                    step+=1

                stats.write("]")