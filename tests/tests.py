import sys
sys.path.append("..")
from training.utils import GAE_rewards
import torch
import numpy as np
t = [torch.FloatTensor([1]), torch.FloatTensor([1]), torch.FloatTensor([1])]
# print(t)
print(GAE_rewards([1,2,3], t,0.9,0.9))

# gamma = 0.9
# lamda = 0.9
# value_old_state = [1,1,1]
# value_new_state = [1,1,1]
# reward = [1,2,3]
# done = [0,0,1]

# batch_size = 3

# advantage = np.zeros(batch_size + 1)

# for t in reversed(range(batch_size)):
#     delta = reward[t] + (gamma * value_new_state[t] * done[t]) - value_old_state[t]
#     advantage[t] = delta + (gamma * lamda * advantage[t + 1] * done[t])

# value_target = advantage[:batch_size] + np.squeeze(value_old_state)

# print(advantage[:batch_size], value_target)