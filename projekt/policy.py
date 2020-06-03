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
      self.model  = baseline.Recurrent(config)
      self.config = config

   #Initial hidden state for RLlib Trainer
   def get_initial_state(self):
      return [self.model.valueF.weight.new(1, self.config.HIDDEN).zero_(),
              self.model.valueF.weight.new(1, self.config.HIDDEN).zero_()]

   def forward(self, input_dict, state, seq_lens):
      logitDict, state = self.model(input_dict['obs'], state, seq_lens)

      logits = []
      #Flatten structured logits for RLlib
      for atnKey, atn in sorted(self.space.items()):
         for argKey, arg in sorted(atn.spaces.items()):
            logits.append(logitDict[atnKey][argKey])

      return torch.cat(logits, dim=1), state

   def value_function(self):
      return self.model.value

   def attention(self):
      return self.model.attn

