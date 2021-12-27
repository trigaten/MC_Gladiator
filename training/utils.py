import torch
import numpy as np

def GAE_adv(rewards, values, DISCOUNT, LAMBDA):
    """This method computes GAE advantages"""
    values = [value.item() for value in values]
    tds = np.empty(len(values), dtype=object)
    # calculate td errors at each position
    for index, reward in enumerate(rewards):
        tds[index] = reward + DISCOUNT * (values[index+1] if index+1 < len(values) else 0) - values[index]
    advs = torch.FloatTensor(len(values))
    # set last advantage equal to last td-0 value
    advs[len(advs)-1] = tds[len(tds)-1]
    # calculate advantages
    for index, td in enumerate(reversed(tds[:-1]), start=1):
        if len(advs) - index - 1 < 0:
            break
        advs[len(advs) - index - 1] = DISCOUNT * LAMBDA * advs[len(advs) - index] + td

    return advs


