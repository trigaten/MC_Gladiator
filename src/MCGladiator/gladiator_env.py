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
from src.MCGladiator.player_client import HumanSurvivalMultiplayer
import os
import pickle

import json

from typing import List, Dict

# first spin up minecraft server in screen


class GladiatorEnv:
    def __init__(
        self,
        agent_names: List[str],
        local_ip: str,
        screen_name: str,
    ):
        """
        
        :param agent_names: list of agent names
        :param local_ip: local ip of the server (usually 127.0.0.1:25565)
        :param screen_name: name of the Linux screen in which the Minecraft server is running (usually "mc")
        """
        self.server = Server(screen_name)
        self.local_ip = local_ip
        self.clients = self.start_clients(agent_names)

        self.last_healths = {agent_name:20 for agent_name in agent_names}


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

        for agent_name, client in self.clients.items():
            obs, _, _, _ = client.step({})
            multi_obs[agent_name] = obs

        multi_obs = self.deal_with_locations(multi_obs)

        return multi_obs, {}

    def step(self, actions:Dict, render:bool=False):
        multi_obs = {}
        multi_dones = {}
        multi_infos = {}

        for agent_name, action in actions.items():
            obs, _, done, info = self.clients[agent_name].step(action)
            multi_obs[agent_name] = obs
            multi_dones[agent_name] = done
            multi_infos[agent_name] = info
            if render:
                self.clients[agent_name].render()

        


        multi_obs = self.deal_with_locations(multi_obs)

        multi_rewards = {agent_name:0 for agent_name in self.clients.keys()}
        multi_rewards, multi_dones = self.deal_with_rewards(multi_obs, multi_dones, multi_rewards)

        done = True in multi_dones.values()
        # TODO: deal with terminated/truncated
        return multi_obs, multi_rewards, done, done, multi_infos

    def deal_with_locations(self, multi_obs):
        agent_names = list(self.clients.keys())
        def _deal_with_locations_helper(first, second):
            first_obs = multi_obs[first]
            second_obs = multi_obs[second]
            second_loc = second_obs["location_stats"]
            first_obs["enemy_loc"] = (second_loc["xpos"], second_loc["ypos"], second_loc["zpos"])
            first_obs["own_loc"] = (first_obs["location_stats"]["xpos"], first_obs["location_stats"]["ypos"], first_obs["location_stats"]["zpos"])
        
        _deal_with_locations_helper(agent_names[0], agent_names[1])
        _deal_with_locations_helper(agent_names[1], agent_names[0])

        # remove location stats
        # for agent_name in agent_names:
        #     multi_obs[agent_name].pop("location_stats")

        return multi_obs

    def deal_with_rewards(self, multi_obs, multi_dones, multi_rewards):
        """Only reward based on damage done for now."""
        agent_names = list(self.clients.keys())
        def _deal_with_rewards_helper(first, second):
            second_health = multi_obs[second]["life_stats"]["life"]
            multi_rewards[first] += max(self.last_healths[second] - second_health, 0)
            if second_health == 20 and self.last_healths[second] < 10:
                multi_dones[second] = True
            
        _deal_with_rewards_helper(agent_names[0], agent_names[1])
        _deal_with_rewards_helper(agent_names[1], agent_names[0])

        for agent_name, obs in multi_obs.items():
            self.last_healths[agent_name] = obs["life_stats"]["life"]

        return multi_rewards, multi_dones
    def render(self):
        for client in self.clients.values():
            client.render()

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
                