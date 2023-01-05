import math

ANGLE = 10

class KillBot:
    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, obs):

        enemy_x, _, enemy_z = obs["enemy_loc"]

        dx = enemy_x - obs['location_stats']["xpos"]
        dz = enemy_z - obs['location_stats']["zpos"]

        # in -180 to 180
        position_angle = (math.degrees(math.atan2((dz), (dx)))) 
        # in -360 to 360
        yaw = obs['location_stats']["yaw"] % 360

        if abs(yaw) > 180:
            if yaw > 0:
                yaw -= 360
            else:
                yaw += 360

        print(position_angle, yaw)
        if position_angle < -90:
        diff = yaw + 90 - position_angle

        if abs(diff) > ANGLE:
            if diff > 0:
                action = {"camera":[0, -ANGLE]}
            else:
                action = {"camera":[0, ANGLE]}
        else:
            action = {"camera":[0, 0]}
        print(action)
        return action

    def compute_rotation(self, )
