from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical

def classify(logits, mask=None, eps=1e-3):
   '''Sample an action from logits'''
   if len(logits.shape) == 1:
      logits = logits.view(1, -1)

   #Masking the noise required. Do not do this
   #in place. We do not want to -inf pad logits
   #for pg loss, only for Categorical sampling
   logits = logits + eps
   if mask is not None:
      logits[~mask] = -np.inf

   distribution = Categorical(logits=logits)
   atn = distribution.sample()
   return atn

def dot(k, v):
   '''Compute v*kT (transpose on last two dims)'''
   kt = k.transpose(-1, -2)
   x = torch.matmul(v, kt)
   x = x.squeeze(-1)
   return x

