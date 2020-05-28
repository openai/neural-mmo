from pdb import set_trace as T
import numpy as np

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension

import torch
from torch import nn

from forge.ethyr.torch import io
from forge.ethyr.torch import policy
from forge.ethyr.torch.policy import baseline

from projekt.realm import actionSpace

class Policy(RecurrentNetwork):
   def __init__(self, *args, config=None, **kwargs):
      super().__init__(*args, **kwargs)
      self.space  = actionSpace(config).spaces
      self.h      = config.HIDDEN
      self.config = config

      #Attentional IO Networks
      self.input  = io.Input(config, policy.Input,
            baseline.Attributes, baseline.Entities)
      self.output = io.Output(config)

      #Standard recurrent hidden network and fc value network
      self.hidden = nn.LSTM(config.HIDDEN, config.HIDDEN)
      self.valueF = nn.Linear(config.HIDDEN, 1)

   #Initial hidden state for RLlib Trainer
   def get_initial_state(self):
      return [self.valueF.weight.new(1, self.h).zero_(),
              self.valueF.weight.new(1, self.h).zero_()]

   def forward(self, input_dict, state, seq_lens):
      #Attentional input preprocessor and batching
      obs, lookup, self.attn = self.input(input_dict['obs'])
      obs   = add_time_dimension(obs, seq_lens, framework="torch")
      batch = obs.size(0)
      h, c  = state

      #Hidden Network and associated data transformations.
      #Pytorch (seq_len, batch, hidden); RLlib (batch, seq_len, hidden)
      #Optimizers batch over traj segments; Rollout workers use seq_len=1
      obs        = obs.view(batch, -1, self.h).transpose(0, 1)
      h          = h.view(batch, -1, self.h).transpose(0, 1)
      c          = c.view(batch, -1, self.h).transpose(0, 1)
      obs, state = self.hidden(obs, [h, c])
      obs        = obs.transpose(0, 1).reshape(-1, self.h)
      state      = [state[0].transpose(0, 1), state[1].transpose(0, 1)]

      #Structured attentional output postprocessor and value function
      logitDict  = self.output(obs, lookup)
      self.value = self.valueF(obs).squeeze(1)
      logits     = []

      #Flatten structured logits for RLlib
      for atnKey, atn in sorted(self.space.items()):
         for argKey, arg in sorted(atn.spaces.items()):
            logits.append(logitDict[atnKey][argKey])

      return torch.cat(logits, dim=1), state

   def value_function(self):
      return self.value

