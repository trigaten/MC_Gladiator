"""

Adapted from https://github.com/minerllabs/minerl/blob/dev/tests/multiagent_test.py
"""
from minerl.env.malmo import InstanceManager
import argparse
from pvpbox_specs import PvpBox

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

class PvpBoxMultiAgent(PvpBox):
    def create_server_quit_producers(self):
        return []

if __name__ == '__main__':

    # Two agents in the same world!
    env_spec = PvpBoxMultiAgent(agent_count=2)

    # IF you want to use existing instances use this!
    # instances = [
    #     InstanceManager.add_existing_instance(9001),
    #     InstanceManager.add_existing_instance(9002)]
    instances = []

    env = env_spec.make(instances=instances)
    
    # make the agents move and stuff
    # iterate desired episodes
    for r in range(10):
        env.reset()
        steps = 0

        done = False
        actor_names = env.task.agent_names
        while not done:
            steps += 1
            env.render()

            actions = env.action_space.no_op()
            for agent in actions:
                actions[agent]["forward"] = 0.1
                actions[agent]["attack"] = 1
                actions[agent]["camera"] = [0, 0.1]
            # actions = env.action_space.sample()

            # print(str(steps) + " actions: " + str(actions))

            obs, reward, done, info = env.step(actions)
            
            print(obs)
            # log("reward: " + str(reward))
            # log("done: " + str(done))
            # log("info: " + str(info))
            # log(" obs: " + str(obs))
