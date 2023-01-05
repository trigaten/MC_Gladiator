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

from typing import List, Dict

# first spin up minecraft server in screen


class GladiatorEnv:
    def __init__(
        self,
        agent_names: List[str] = ["bot1", "bot2"],
        local_ip: str = "127.0.0.1:25565",
        screen_name: str = "mc",
    ):
        self.server = Server(screen_name)
        self.local_ip = local_ip
        self.clients = self.start_clients(agent_names)


    def start_clients(self, agent_names):
        clients = {}
        for agent in agent_names:
            client = HumanSurvivalMultiplayer(self.local_ip, agent).make()
            clients[agent] = client
            client.reset()

        return clients

    def reset(self):
        server = self.server
        server.execute("clear @a")
        tp_locations = ["-2.5 5 0.5 270 0", "3.5 5 0.5 90 0"]
        for agent_name, tp_location in zip(list(self.clients.keys()), tp_locations):
            server.execute(f"tp {agent_name} {tp_location}")


        server.execute("gamerule naturalRegeneration false")
        server.execute("clear @a")
        server.execute("give @a minecraft:iron_sword 1")
        server.execute("effect give @a instant_health 1 200")
        server.execute("effect give @a saturation 1 255")
        time.sleep(0.1)

        multi_obs = {}

        for client in self.clients.values():
            obs, _, _, _ = client.step()
            multi_obs[client.agent_name] = obs

        return multi_obs

    def step(self, actions:Dict, render:bool=False):
        multi_obs = {}
        multi_rewards = {}
        multi_dones = {}
        multi_infos = {}

        for agent_name, action in actions.items():
            obs, reward, done, info = self.clients[agent_name].step(action)
            multi_obs[agent_name] = obs
            multi_rewards[agent_name] = reward
            multi_dones[agent_name] = done
            multi_infos[agent_name] = info
            if render:
                self.clients[agent_name].render()

        return multi_obs, multi_rewards, multi_dones, multi_infos

    # def start_training(self, episodes:int=1000):
    #     for i in range(episodes):
    #         self.reset()
    #         done = False
    #         if not os.path.exists("episodes/" + str(i)):
    #             os.mkdir("episodes/" + str(i))
    #         while not done:
    #             # compute actions for each bot
    #             for agent_dict in self.agent_info_dicts:
    #                 agent = agent_dict["agent"]
    #                 agent_dict["action"] = agent(agent_dict["last_obs"])

    #             # execute actions
    #             for agent_dict in self.agent_info_dicts:
    #                 action = agent_dict["action"]
    #                 obs, reward, done, info = agent_dict["client"].step(action)
    #                 agent_dict["agent"].push(agent_dict["last_obs"], action, reward, done, obs)
    #                 agent_dict["last_obs"] = obs
                