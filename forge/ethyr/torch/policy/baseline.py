from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn.utils import rnn

from forge.blade.io.stimulus.static import Stimulus

from forge.ethyr.torch import policy
from forge.ethyr.torch import io

class Base(nn.Module):
   def __init__(self, config):
      '''Base class for baseline policies

      Args:
         config: A Configuration object
      '''
      super().__init__()
      self.embed  = config.EMBED
      self.config = config

      self.output = io.Output(config)
      self.input  = io.Input(config,
            embeddings=policy.BiasedInput,
            attributes=policy.Attention)

      self.valueF = nn.Linear(config.HIDDEN, 1)

   def hidden(self, obs, state=None, lens=None):
      '''Abstract method for hidden state processing, recurrent or otherwise,
      applied between the input and output modules

      Args:
         obs: An observation dictionary, provided by forward()
         state: The previous hidden state, only provided for recurrent nets
         lens: Trajectory segment lengths used to unflatten batched obs
      ''' 
      raise NotImplementedError('Implement this method in a subclass')

   def forward(self, obs, state=None, lens=None):
      '''Applies builtin IO and value function with user-defined hidden
      state subnetwork processing. Arguments are supplied by RLlib
      ''' 
      entityLookup  = self.input(obs)
      hidden, state = self.hidden(entityLookup, state, lens)
      self.value    = self.valueF(hidden).squeeze(1)
      actions       = self.output(hidden, entityLookup)
      return actions, state

class Simple(Base):
   def __init__(self, config):
      '''Simple baseline model with flat subnetworks'''
      super().__init__(config)
      h = config.HIDDEN

      self.conv   = nn.Conv2d(h, h, 3)
      self.pool   = nn.MaxPool2d(2)
      self.fc     = nn.Linear(h*3*3, h)

      self.proj   = nn.Linear(2*h, h)
      self.attend = policy.Attention(self.embed, h)

   def hidden(self, obs, state=None, lens=None):
      #Attentional agent embedding
      agents, _ = self.attend(obs[Stimulus.Entity])

      #Convolutional tile embedding
      tiles     = obs[Stimulus.Tile]
      self.attn = torch.norm(tiles, p=2, dim=-1)

      w      = self.config.WINDOW
      batch  = tiles.size(0)
      hidden = tiles.size(2)
      #Dims correct?
      tiles  = tiles.reshape(batch, w, w, hidden).permute(0, 3, 1, 2)
      tiles  = self.conv(tiles)
      tiles  = self.pool(tiles)
      tiles  = tiles.reshape(batch, -1)
      tiles  = self.fc(tiles)

      hidden = torch.cat((agents, tiles), dim=-1)
      hidden = self.proj(hidden)
      return hidden, state

class Recurrent(Simple):
   def __init__(self, config):
      '''Recurrent baseline model'''
      super().__init__(config)
      self.lstm   = nn.LSTM(config.HIDDEN, config.HIDDEN)

   #Note: seemingly redundant transposes are required to convert between 
   #Pytorch (seq_len, batch, hidden) <-> RLlib (batch, seq_len, hidden)
   def hidden(self, obs, state, lens):
      #Attentional input preprocessor and batching
      hidden, _ = super().hidden(obs)
      config    = self.config
      batch     = len(lens)
      h, c      = state

      #Pack (padded_seq x batch, hidden) -> (padded_seq, batch, hidden)
      hidden        = hidden.view(batch, -1, config.HIDDEN).transpose(0, 1)
      hidden        = rnn.pack_padded_sequence(
                        hidden, lens, enforce_sorted=False)

      #Permute (batch, 1?, hidden) -> (1, batch, hidden)
      h             = h.view(batch, -1, config.HIDDEN).transpose(0, 1)
      c             = c.view(batch, -1, config.HIDDEN).transpose(0, 1)

      #Main recurrent network
      hidden, state = self.lstm(hidden, [h, c])

      #Permute (1, batch, hidden) -> (batch, 1, hidden)
      h             = state[0].transpose(0, 1)
      c             = state[1].transpose(0, 1)

      #Unpack (padded_seq, batch, hidden) -> (padded_seq x batch, hidden)
      hidden, _     = rnn.pad_packed_sequence(hidden)
      hidden        = hidden.transpose(0, 1).reshape(-1, config.HIDDEN)

      return hidden, [h, c]

class Attentional(Base):
   def __init__(self, config):
      '''Transformer-based baseline model'''
      super().__init__(config)
      self.agents = nn.TransformerEncoderLayer(d_model=config.HIDDEN, nhead=8)
      self.tiles  = nn.TransformerEncoderLayer(d_model=config.HIDDEN, nhead=8)
      self.proj   = nn.Linear(2*config.HIDDEN, config.HIDDEN)

   def hidden(self, obs, state=None, lens=None):
      #Attentional agent embedding
      agents    = self.agents(obs[Stimulus.Entity])
      agents, _ = torch.max(agents, dim=-2)

      #Attentional tile embedding
      tiles     = self.tiles(obs[Stimulus.Tile])
      self.attn = torch.norm(tiles, p=2, dim=-1)
      tiles, _  = torch.max(tiles, dim=-2)

      
      hidden = torch.cat((tiles, agents), dim=-1)
      hidden = self.proj(hidden)
      return hidden, state
