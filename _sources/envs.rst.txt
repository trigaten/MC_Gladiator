Environments
========================

Here are a few training environments.

All environments should be instantiated as 

.. code:: python

    env = EnvName(agent_count=2).make(instances=[])


`instances` can also be a list of live MineRL instances.


.. autoclass:: mcgladiator.envs.pvpbox_specs.PvpBox
