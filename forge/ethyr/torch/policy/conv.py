from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

def Conv2d(fIn, fOut, k, stride=1):
   '''torch Conv2d with same padding'''
   assert k % 2 == 0
   pad = int((k-1)/2)
   return torch.nn.Conv2d(fIn, fOut, 
      k, stride=stride, padding=pad)

def Pool(k, stride=1, pad=0):
   return torch.nn.MaxPool2d(
      k, stride=stride, padding=pad)

class ConvReluPool(nn.Module):
   def __init__(self, fIn, fOut, 
         k, stride=1, pool=2):
      super().__init__()
      self.conv = Conv2d(
         fIn, fOut, k, stride)
      self.pool = Pool(k)

   def forward(self, x):
      x = self.conv(x)
      x = F.relu(x)
      x = self.pool(x)
      return x

