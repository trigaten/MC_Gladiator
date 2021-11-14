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
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', action="store_true", help='use the single agent default xml')
    parser.add_argument('--port', type=int, default=None, help='the port of existing client or None to launch')
    parser.add_argument('--episodes', type=int, default=2, help='the number of resets to perform - default is 1')
    args = parser.parse_args()

    # logs
    import coloredlogs
    import logging

    coloredlogs.install(level=logging.DEBUG)

    # clear logs
    import subprocess

    logging.debug("Deleting previous java log files...")
    subprocess.check_call("rm -rf logs/*", shell=True)

    # Two agents in the same world!
    if args.single:
        env_spec = PvpBoxMultiAgent(agent_count=1)
    else:
        env_spec = PvpBoxMultiAgent(agent_count=2)

    # IF you want to use existing instances use this!
    # instances = [
    #     InstanceManager.add_existing_instance(9001),
    #     InstanceManager.add_existing_instance(9002)]
    instances = []

    env = env_spec.make(instances=instances)
    
    # make the agents move and stuff
    # iterate desired episodes
    for r in range(args.episodes):
        logging.debug(f"Reset for episode {r + 1}")
        env.reset()
        steps = 0

        done = False
        actor_names = env.task.agent_names
        while not done:
            steps += 1
            env.render()

            actions = env.action_space.no_op()
            for agent in actions:
                actions[agent]["forward"] = 1
                actions[agent]["attack"] = 1
                actions[agent]["camera"] = [0, 0.1]

            # print(str(steps) + " actions: " + str(actions))

            obs, reward, done, info = env.step(actions)

            # log("reward: " + str(reward))
            # log("done: " + str(done))
            # log("info: " + str(info))
            # log(" obs: " + str(obs))

        logging.debug(f"Episode {r + 1}/{args.episodes} done: {steps} steps")
