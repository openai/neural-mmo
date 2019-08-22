from pdb import set_trace as T
import numpy as np
import torch
import time

from collections import defaultdict

import projekt
from forge.ethyr.torch import save
from forge.ethyr.torch import Model
from forge.ethyr.torch.model import PopulationOptimizer, GradientOptimizer
from forge.blade.lib.log import Quill, BlobLogs

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
      if config.POPOPT:
         self.opt = PopulationOptimizer(self.net, config)
      else:
         self.opt = GradientOptimizer(self.net, config)

      if config.LOAD or config.BEST:
         self.net.load(self.opt, config.BEST)

      self.quill = Quill(config.MODELDIR)
      self.log = defaultdict(list)

      self.tick = 0
      self.net.nParams

   @runtime
   def step(self):
      '''Broadcasts updated weights to server level
      God optimizer nodes. Performs an Adam step
      once optimizers return a batch of gradients.''' 
      
      recvs = super().step(self.net.weights)

      #Write logs using Quill
      recvs, logs = list(zip(*recvs))
      logs        = BlobLogs.merge(logs)

      self.quill.scrawl(logs)
      self.tick += 1

      self.quill.print()
      if not self.config.TEST:
         lifetime = self.quill.latest()
         self.opt.step(recvs, logs)
         self.net.checkpoint(self.opt, lifetime)
