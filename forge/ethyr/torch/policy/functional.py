from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical

def classify(logits, mask=None):
   '''Sample an action from logits'''
   if len(logits.shape) == 1:
      logits = logits.view(1, -1)

   #Masking the noise required
   logits += 1e-3
   if mask is not None:
      logits[1-mask] = -np.inf

   #distribution = Categorical(torch.softmax(logits, dim=1))
   distribution = Categorical(logits=logits)
   atn = distribution.sample()
   return atn

def dot(k, v):
   '''Compute v*kT (transpose on last two dims)'''
   kt = k.transpose(-1, -2)
   x = torch.matmul(v, kt)
   x = x.squeeze(-1)
   return x

