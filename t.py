from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
import gym
env = HumanSurvival(resolution=(640, 360)).make()

print(env.reset())