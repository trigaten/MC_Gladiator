import math

ANGLE = 10

class KillBot:
    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, obs):

        enemy_x, _, enemy_z = obs["enemy_loc"]

        return self.compute_rotation((obs['location_stats']["xpos"], obs['location_stats']["zpos"]), (enemy_x, enemy_z), obs['location_stats']["yaw"])


    def compute_rotation(self, pos, enemy_pos, yaw):
        dx = enemy_pos[0] - pos[0]
        dz = enemy_pos[1] - pos[1]

        # in -180 to 180
        position_angle = (math.degrees(math.atan2((dz), (dx)))) 
        # turn yaw into -180 to 180
        restricted_yaw = yaw % 360
        if abs(restricted_yaw) > 180:
            if restricted_yaw > 0:
                restricted_yaw -= 360
            else:
                restricted_yaw += 360

        # if position_angle < -90:
        #     position_angle = -position_angle

        diff = restricted_yaw + 90 - position_angle

        if abs(diff) > ANGLE:
            if diff > 0:
                action = {"camera":[0, -ANGLE]}
            else:
                action = {"camera":[0, ANGLE]}
        else:
            action = {"camera":[0, 0]}

        return action