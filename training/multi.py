from minerl.env.malmo import InstanceManager
import sys
sys.path.append("..")
from env.pvpbox_specs import PvpBox
import gym
import minerl  # noqa
import argparse
import time
from env.OneVersusOneWrapper import OneVersusOneWrapper

class TreechopMultiAgentNoQuit(PvpBox):
    # This version of treechop doesn't terminate the episode 
    # if the other agent quits/dies (or gets the max reward)
    # def create_server_quit_producers(self):
    #     return []
    pass

actions = ["attack", "turn_left", "turn_right"]    

if __name__ == '__main__':
    env_spec = TreechopMultiAgentNoQuit(agent_count=2)

    # IF you want to use existing instances use this!
    # instances = [
    #     InstanceManager.add_existing_instance(9001),
    #     InstanceManager.add_existing_instance(9002)]
    instances = []

    env = OneVersusOneWrapper(env_spec.make(instances=instances))

    # iterate desired episodes
    while True:
        env.reset()
        steps = 0

        done = False
        actor_names = env.env.task.agent_names
        while not done:
            steps += 1
            env.render()

            actions = env.env.action_space.no_op()
            for agent in actions:
                actions[agent]["forward"] = 1
                actions[agent]["attack"] = 1
                actions[agent]["camera"] = [0, 0.1]

            obs, reward, done, info = env.step(actions)
            print(reward)


