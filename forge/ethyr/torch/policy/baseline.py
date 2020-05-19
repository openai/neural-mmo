import numpy as np
from pdb import set_trace as T

import torch
from torch import nn

from forge.ethyr.torch import policy
from forge.ethyr.torch import io

class IO(nn.Module):
    def __init__(self, config):
      '''Input and output networks

      Args:                                                                   
         config: A Configuration object
      '''
      super().__init__()
      self.input  = io.Input(config, policy.Input, Attributes, Entities)
      self.output = io.Output(config)

class Attributes(policy.Attention):
    def __init__(self, config):
      '''Attentional network over attributes

      Args:
         config: A Configuration object
      '''
      super().__init__(config.EMBED, config.HIDDEN)

class Entities(nn.Module):
   def __init__(self, config):
      '''Attentional network over entities

      Args:
         config: A Configuration object
      '''
      super().__init__()
      h = config.HIDDEN
      self.h = h
      #self.targDim = 250*h

      self.conv = nn.Conv2d(h, h, 3)
      self.pool = nn.MaxPool2d(2)
      self.fc1 = nn.Linear(h*3*3, h)

      self.fc2  = nn.Linear(2*h, h)
      self.attn = policy.Attention(config.EMBED, config.HIDDEN)

   def forward(self, x):
      batch = x.shape[0]
      conv  = x[:, -81:]

      scores = torch.norm(x, p=2, dim=-1).squeeze(0)

      conv = conv.view(-1, 9, 9, self.h).permute(0, 3, 1, 2)
      conv = self.conv(conv)
      conv = self.pool(conv)
      conv = conv.reshape(batch, -1)
      conv = self.fc1(conv)

      attn = x[:, :-81]
      attn, _ = self.attn(attn)

      x = torch.cat((attn, conv), dim=-1)
      x = self.fc2(x)

      #x = x.view(-1)
      #pad = torch.zeros(self.targDim - len(x)).to(self.device)
      #x = torch.cat([x, pad])
      #x = self.fc(x)
      return x, scores

