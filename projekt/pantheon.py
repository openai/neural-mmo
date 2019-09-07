from pdb import set_trace as T
import numpy as np
import torch
import time, os
import ray

from collections import defaultdict

import projekt
from projekt.timed import Summary

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
      self.log = defaultdict(list)

      self.net.nParams

   @runtime
   def tick(self):
      '''Inner timed step'''
      #self.resetLogs()
      recvs = super().step(self.net.weights)

      #Write logs using Quill
      recvs, blobs, log = list(zip(*recvs))
      blobs = BlobSummary.merge(blobs)

      self.net.step(recvs, blobs, log)

      return log


   def step(self):
      '''Broadcasts updated weights to server level
      God optimizer nodes. Performs an Adam step
      once optimizers return a batch of gradients.''' 
      log = self.tick()
      
      stats = self.net.quill.stats()
      log = Log.summary([self.discipleLogs(), *log, self.logs()])
      log = str(Summary(log))
  
      path = os.path.join(self.config.MODELDIR, 'stats.txt')
      with open(path, 'a') as f:
         f.write(stats)
         f.write(log)

      return stats, log
