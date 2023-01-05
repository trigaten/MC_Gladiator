import math

ANGLE = 10

class KillBot:
    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, obs):

        enemy_x, _, enemy_z = obs["enemy_loc"]

        dx = enemy_x - obs['location_stats']["xpos"]
        dz = enemy_z - obs['location_stats']["zpos"]

        distance = math.sqrt(dx**2 + dz**2)

        return {"camera":[0, position_angle - yaw]}
