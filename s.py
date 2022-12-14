class HumanSurvivalMultiplayer(HumanSurvival):
    def __init__(self, server_ip, player_name, *args, **kwargs):
        self.server_ip = server_ip
        self.player_name = player_name
        super().__init__(
            *args, **kwargs
        )

    def create_observables(self) -> List[Handler]:
        # To make this bit faster (full obs stuff is slow)
        return [handlers.POVObservation(self.resolution)]

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
