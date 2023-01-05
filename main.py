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

import json

# first spin up minecraft server in screen


class training_instance:
    def __init__(
        self,
        agents: List[str] = ["bot1", "bot2"],
        local_ip: str = "127.0.0.1:25565",
        server_str: str = "mc",
    ):
        self.server = Server(server_str)
        self.local_ip = local_ip
        self.agent_info_dicts = self.make_agent_info_dicts(agents)


    def make_agent_info_dicts(self, agents):
        dicts = []
        for agent in agents:
            agent_dict = {}
            agent_dict["client"] = HumanSurvivalMultiplayer(self.local_ip, str(agent)).make()
            agent_dict["agent"] = agent
            agent_dict["last_obs"] = agent_dict["client"].reset()
            dicts.append(agent_dict)

        return dicts

    def reset(self):
        server = self.server
        server.execute("clear @a")
        tp_locations = ["-2.5 5 0.5 270 0", "3.5 5 0.5 90 0"]
        for agent_dict, tp_location in zip(self.agent_info_dicts, tp_locations):
            server.execute(f"tp {str(agent_dict['agent'])} {tp_location}")
            agent_dict["last_obs"] = agent_dict.client.step()
            agent_dict["action"] = None

        server.execute("gamerule naturalRegeneration false")
        server.execute("clear @a")
        server.execute("give @a minecraft:iron_sword 1")
        server.execute("effect give @a instant_health 1 200")
        server.execute("effect give @a saturation 1 255")
        time.sleep(0.1)

    def start_training(self, episodes:int=1000):
        for i in range(episodes):
            self.reset()
            done = False
            if not os.path.exists("episodes/" + str(i)):
                os.mkdir("episodes/" + str(i))
            while not done:
                # compute actions for each bot
                for agent_dict in self.agent_info_dicts:
                    agent = agent_dict["agent"]
                    agent_dict["action"] = agent(agent_dict["last_obs"])

                # execute actions
                for agent_dict in self.agent_info_dicts:
                    action = agent_dict["action"]
                    obs, reward, done, info = agent_dict["client"].step(action)
                    agent_dict["agent"].push(agent_dict["last_obs"], action, reward, done, obs)
                    agent_dict["last_obs"] = obs
                