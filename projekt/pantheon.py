from pdb import set_trace as T
import numpy as np
import torch
import time

from collections import defaultdict

import projekt
from forge.ethyr.torch import save
from forge.ethyr.torch import Model
from forge.blade.lib.log import Quill

from forge import trinity
from forge.trinity.timed import runtime

class Pantheon(trinity.Pantheon):
   '''Cluster level Pantheon API demo

   This cluster level module aggregrates
   gradients across all server level optimizer
   nodes and updates model weights using Adam.

   Also demonstrates logging and snapshotting
   functionality through the Quill and Model
   libraries, respectively.'''

   def __init__(self, trinity, config, args):
      '''Initializes a copy of the model, which keeps
      track of a copy of the weights for the optimizer.'''
      super().__init__(trinity, config, args)      
      self.config, self.args = config, args

      self.net = Model(projekt.ANN, config, args)
      self.quill = Quill(config.MODELDIR)
      self.log = defaultdict(list)

      self.tick = 0
      self.net.nParams

   @runtime
   def step(self):
      '''Broadcasts updated weights to server level
      God optimizer nodes. Performs an Adam step
      once optimizers return a batch of gradients.''' 
      
      recvs = super().step(self.net.model)

      #Write logs using Quill
      recvs, logs = list(zip(*recvs))
      self.quill.scrawl(logs)
      self.tick += 1

      if self.config.TEST:
         self.quill.print()
      else:
         lifetime = self.quill.latest()
         self.net.stepOpt(recvs)
         self.net.checkpoint(lifetime)
         self.net.saver.print()
