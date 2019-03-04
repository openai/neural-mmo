from pdb import set_trace as T
import numpy as np
import ray
import torch
from torch import nn


@ray.remote
def foo(i):
   np.random.seed(i)
   torch.manual_seed(i)
   fc = nn.Sequential(nn.Linear(2, 2))
   print(i, ': ', [e for e in fc.parameters()][0])

ray.init()
jobs = [foo.remote(i) for i in range(5)]
print(ray.get(jobs))


