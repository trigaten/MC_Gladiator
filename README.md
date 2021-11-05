# GLADIATOR-Project
Training a superhuman sword fighting agent with Reinforcement Learning and Project Malmo

## How to make a custom environment (by copying a preexisting one)

Clone MineRL

`git clone https://github.com/minerllabs/minerl.git`

Make a new Jupyter Notebook inside of the cloned repo

### Create a copy of the environment

Navigate to `minerl/minerl/herobraine/env_specs`

Make a copy of `treechop_specs.py` and rename it to `treechop_specs2.py`

Open `treechop_specs2.py` and change the class name (Line 121) to "Treechop2". 

Change the kwargs['name'] line (line 124) to "MineRLTreechop2-v0".

Navigate to `minerl/minerl/herobraine` and open `envs.py`

In the imports section (around line 11) add the import for our new environment `from minerl.herobraine.env_specs.treechop_specs2 import Treechop2
`
After the definition of MINERL_TREECHOP_V0, add `MINERL_TREECHOP2_V0 = Treechop2()`

Add `MINERL_TREECHOP2_V0` to the comp_envs array (around line 51)

Also add it to the BASIC_ENV_SPECS array (around line 123)

### Using our new environment

In the Notebook, put this code
```import gym
from minerl.herobraine.env_specs.treechop_specs2 import Treechop2
x = Treechop2()
gym.envs.register(
     id='MineRLTreeChop2-v0',
     entry_point='minerl.herobraine.env_specs.treechop_specs2.py',
     max_episode_steps=1000,
)
ENV_SPEC_MAPPINGS = {}
# x.register()
ENV_SPEC_MAPPINGS[x.name] = x
env = gym.make(x.name)

obs = env.reset()```

## Model

We will be using a multi headed model with a CNN+GRU+FC base and two FC heads.


### https://adoptium.net/archive.html?variant=openjdk8&jvmVariant=hotspot
