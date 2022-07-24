
import torch
import torch.nn as nn
import configparser
print("DEVICES", torch.cuda.device_count())
print("__________________________")
config = configparser.RawConfigParser()
config.read("ray_config.cfg")


paths = dict(config.items('PATHS'))

import os
import sys
sys.path.append(os.getcwd()) 


sys.path.append("..")
sys.path.append(paths["glob"])
import ray
ray.init(runtime_env={"working_dir": paths["work"]})
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import dqn
from gym.wrappers import TimeLimit

from ray.rllib.models.catalog import MODEL_DEFAULTS

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, same_padding, \
    SlimConv2d, SlimFC
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from environment.dummy_spec import DummyGym
from environment.dummy_spec import DummyMAGym
from environment.pvpbox_specs import PvpBox
from environment.wrappers import *

from ray import tune

agent_actions = [("attack", 1), ("left", 1), ("back", 1), ("right", 1), ("forward", 1), ("camera", [0,15]), ("camera", [0,-15])]
num_actions = len(agent_actions)
def env_creator(env_config):
    return OneVersusOneWrapper(SuperviserWrapper(PvpBox(agent_count=2).make(instances=[])), agent_actions)
            

register_env("1v1env", env_creator)
from DQN import DeepQNet

ModelCatalog.register_custom_model("cnet", DeepQNet)
tune.run(
    "DQN",
    name="MINE_DQN",
    stop={"episodes_total": 60000},
    checkpoint_freq=100,
    local_dir="ray_out",
    config={
        # Enviroment specific
        "env": "1v1env",
        "framework":"torch",
        # General
        # "log_level": "ERROR",
        "num_gpus": 2,
        "num_workers": 2,
        # "num_workers": 0,
        # "num_envs_per_worker": 4,
        "learning_starts": 1000,
        "buffer_size": int(1e5),
        "compress_observations": True,
        # "sample_batch_size": 20,
        "train_batch_size": 256,
        "gamma": .99,
        "hiddens":[],
        # Method specific
        "dueling": False,
        "double_q": False,
        "model": {
            "custom_model": "cnet",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {},
        },
        "multiagent": {
            "policies": {
                "policy_01": (None, DummyMAGym(len(agent_actions)).observation_space, DummyMAGym(len(agent_actions)).action_space, {}),
                "policy_02": (None, DummyMAGym(len(agent_actions)).observation_space, DummyMAGym(len(agent_actions)).action_space, {}),
            },
            "policy_mapping_fn": lambda agent_id:
                "policy_01" if agent_id == "agent_0" else "policy_02",

            "policies_to_train": ["policy_01"]
        },
    },
)

