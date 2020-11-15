'''Observation processing module'''

from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

from forge.blade.io import stimulus, action
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.io.stimulus import node

class Input(nn.Module):
   def __init__(self, config, embeddings, attributes):
      '''Network responsible for processing observations

      Args:
         config     : A configuration object
         embeddings : An attribute embedding module
         attributes : An attribute attention module
         entities   : An entity attention module
      '''
      super().__init__()

      self.embeddings = nn.ModuleDict()
      self.attributes = nn.ModuleDict()

      for _, entity in stimulus.Static:
         continuous = len([e for e in entity if e[1].CONTINUOUS])
         discrete   = len([e for e in entity if e[1].DISCRETE])
         #self.attributes[entity.__name__] = attributes(config.EMBED, config.HIDDEN)
         self.attributes[entity.__name__] = nn.Linear(
               (continuous+discrete)*config.HIDDEN, config.HIDDEN)
         self.embeddings[entity.__name__] = embeddings(
               continuous=continuous, discrete=4096, config=config)

   def forward(self, inp):
      '''Produces tensor representations from an IO object

      Args:                                                                   
         inp: An IO object specifying observations                      
         

      Returns:
         observationTensor : A fixed size observation representation
         entityLookup      : A fixed size representation of each entity
      ''' 
      #Pack entities of each attribute set
      entityLookup = {}

      egocentric = {
         'Tile': {
            'Continuous': (2, 3),
            'Discrete':   (1, 2)
         },
         #'Entity': {
         #'Continuous': (2, 3),
         #'Discrete':   (2, 3)
         #},
      }

      if False in (inp['Entity']['Discrete'].sum(1) >= 0):
         T()

      for entity, dtypes in egocentric.items():
         entities = inp[entity]
         for dtype, idxs in dtypes.items():
            typed             = entities[dtype]
            cent              = typed[:, 40:41, idxs]
            typed[:, :, idxs] = cent - typed[:, :, idxs]

      #Changes prevrun: Hacked discrete egocentric,
      #Removed reordering of center obs (self) to first
      #Added 112 manual indexing in baseline

      #Changed this run: make food/health/water continuous
      #Zero out continuous tile embeddings (was 0.15)
      #Zero out continuous index embedding
      #Set stim=4, 4000 batch
      inp['Tile']['Discrete'][:, :, 1] += 7 + 7
      inp['Tile']['Discrete'][:, :, 2] += 7 + 7 + 8 + 7

      inp['Entity']['Discrete'] *= 0
      tileWeight = torch.Tensor([0.0, 0.0, 1.00, 1.00])
      entWeight  = torch.Tensor([0.0, 0.0, 0.00, 0.00, 0.0, 0.00, 0.1, 0.1, 0.1, 0.0, 0.0, 0.00])

      try:
         inp['Tile']['Continuous']   *= tileWeight
         inp['Entity']['Continuous'] *= entWeight
      except:
         inp['Tile']['Continuous']   *= tileWeight.cuda()
         inp['Entity']['Continuous'] *= entWeight.cuda()
 
      for name, entities in inp.items():
         #Construct: Batch, ents, nattrs, hidden
         embeddings = self.embeddings[name](entities)
         B, N, _, _ = embeddings.shape
         embeddings = embeddings.view(B, N, -1)

         #Construct: Batch, ents, hidden
         #entityLookup[name], scores = self.attributes[name](embeddings)
         entityLookup[name] = self.attributes[name](embeddings)

      return entityLookup
