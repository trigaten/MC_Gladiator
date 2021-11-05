# GLADIATOR-Project
Training a superhuman sword fighting agent with Reinforcement Learning and Project Malmo

## How to make a custom environment (by copying a preexisting one)

Clone MineRL

`git clone https://github.com/minerllabs/minerl.git`

Make a new Jupyter Notebook inside of the cloned repo

### Copy a default environment

Navigate to `minerl/minerl/herobraine/env_specs`

Make a copy of `treechop_specs.py` and rename it to `treechop_specs2.py`

Open `treechop_specs2.py` and change the class name (Line 121) to "Treechop2". 

Change the kwargs['name'] line (line 124) to "MineRLTreechop2-v0".

Change the line with "iron_axe" (around line 140) to "iron_sword".

Navigate to `minerl/minerl/herobraine` and open `envs.py`

In the imports section (around line 11) add the import for our new environment `from minerl.herobraine.env_specs.treechop_specs2 import Treechop2
`
After the definition of MINERL_TREECHOP_V0, add `MINERL_TREECHOP2_V0 = Treechop2()`

Add `MINERL_TREECHOP2_V0` to the comp_envs array (around line 51)

Also add it to the BASIC_ENV_SPECS array (around line 123)

### Using our new environment

In the Notebook, run this code
``` 
import gym
from minerl.herobraine.env_specs.treechop_specs2 import Treechop2
x = Treechop2()
x.register()
env = gym.make(x.name)

obs = env.reset() 
```

Congratulations, you have just made a custom environment and completed the first step of our environment construction-- creating an agent with an iron sword.

This is likely not the most effective way to make a new environment (Im not sure if all of the steps are needed), so I will try to update this when I find better ways.

## Model

We will be using a multi headed model with a CNN+GRU+FC base and two FC heads.


### https://adoptium.net/archive.html?variant=openjdk8&jvmVariant=hotspot
