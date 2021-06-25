from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

from neural_mmo.forge.ethyr.torch import utils

class ReluBlock(nn.Module):
   def __init__(self, h, layers=2, postRelu=True):
      super().__init__()
      self.postRelu = postRelu

      self.layers = utils.ModuleList(
         nn.Linear, h, h, n=layers)

   def forward(self, x):
      for idx, fc in enumerate(self.layers):
         if idx != 0:
            x = torch.relu(x)
         x = fc(x)

      if self.postRelu:
         x = torch.relu(x)

      return x
