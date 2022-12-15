from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from typing import List
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
import coloredlogs
import logging
coloredlogs.install(logging.DEBUG)

class HumanSurvivalMultiplayer(HumanSurvival):
    def __init__(self, server_ip, player_name, *args, **kwargs):
        self.server_ip = server_ip
        self.player_name = player_name
        super().__init__(
            *args, **kwargs
        )

    def create_observables(self) -> List[Handler]:
        # To make this bit faster (full obs stuff is slow)
        return [handlers.POVObservation(self.resolution), handlers.ObservationFromLifeStats()]


    def create_server_initial_conditions(self) -> List[Handler]:
        original_list = super().create_server_initial_conditions()
        return original_list + [
            handlers.RemoteServer(self.server_ip),
        ]

    def create_agent_start(self) -> List[Handler]:
        original_list = super().create_agent_start()
        return original_list + [
            handlers.MultiplayerUsername(self.player_name),
        ]

    def create_monitors(self):
        return []

env = HumanSurvivalMultiplayer("127.0.0.1:25565", "hi2").make()


print(env.reset())

done = False

while not done:
    # Take a random action
    action = env.action_space.sample()
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    action["ESC"] = 0
    obs, reward, done, _ = env.step({"camera":[1,1], "jump":1,"forward":1})
    env.render()
    print(obs["life_stats"])