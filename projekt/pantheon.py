from pdb import set_trace as T
import numpy as np
import torch
import time

from collections import defaultdict

import projekt
from forge.ethyr.torch import save
from forge.ethyr.torch import Model
from forge.ethyr.torch.model import PopulationOptimizer, GradientOptimizer
from forge.blade.lib.log import Quill, BlobSummary


from forge.trinity.ascend import Ascend, runtime, Log

class Pantheon(Ascend):
   '''Cluster level Pantheon API demo

   This cluster level module aggregrates
   gradients across all server level optimizer
   nodes and updates model weights using Adam.

   Also demonstrates logging and snapshotting
   functionality through the Quill and Model
   libraries, respectively.'''

   def __init__(self, trinity, config, idx):
      '''Initializes a copy of the model, which keeps
      track of a copy of the weights for the optimizer.'''
      super().__init__(trinity.god, config.NGOD, trinity, config)
      self.config = config

      self.net = Model(projekt.ANN, config)

      #Have been experimenting with population based
      #training. Nothing stable yet -- advise avoiding
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
      
      #self.resetLogs()
      recvs = super().step(self.net.weights)

      #Write logs using Quill
      recvs, blobs, log = list(zip(*recvs))

      blobs = BlobSummary.merge(blobs)
      self.quill.scrawl(blobs)
      self.tick += 1

      self.quill.print()
      if not self.config.TEST:
         lifetime = self.quill.latest()
         self.opt.step(recvs, blobs)
         self.net.checkpoint(self.opt, lifetime)

      log = Log.summary([self.discipleLogs(), *log])
      return log
