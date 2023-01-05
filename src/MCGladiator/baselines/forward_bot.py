import math

ANGLE = 10

class ForwardBot:
    def __init__(self, name) -> None:
        self.name = name
        self.attack_cooldown=0

    def __call__(self, obs):

        enemy_x, _, enemy_z = obs["enemy_loc"]

        dx = enemy_x - obs['location_stats']["xpos"]
        dz = enemy_z - obs['location_stats']["zpos"]

        distance = math.sqrt(dx**2 + dz**2)

        if distance > 2:
            return {"forward":1}
        else:
            if self.attack_cooldown == 0:
                self.attack_cooldown = 10
                return {"attack":1}
            else:
                self.attack_cooldown -= 1
                return {}
