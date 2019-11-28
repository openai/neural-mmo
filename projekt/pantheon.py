from pdb import set_trace as T
import numpy as np
import time, os
import ray
import torch

from collections import defaultdict

import projekt
from projekt.timed import Summary

from forge.ethyr.torch import save
from forge.ethyr.torch import Model
from forge.ethyr.torch.model import PopulationOptimizer, GradientOptimizer
from forge.blade.lib.log import Quill, BlobSummary

from forge.trinity.ascend import Ascend, runtime, Log

class Pantheon(Ascend):
   '''Cluster level infrastructure demo

   This module aggregates gradients across all
   server level environments and updates model
   weights using Agam.

   Also demonstrates logging and snapshotting
   functionality through the Quill and Model
   libraries, respectively.'''

   def __init__(self, trinity, config, idx):
      '''Initializes a copy of the model, which keeps
      track of the weights for the optimizer.'''
      super().__init__(trinity.god, config.NGOD, trinity, config)
      self.config = config

      self.net = Model(projekt.ANN, config)
      self.log = defaultdict(list)

      self.net.printParams()

   def tick(self):
      recvs             = super().step(self.net.weights)
      recvs, blobs, log = list(zip(*recvs))
      blobs             = BlobSummary.merge(blobs)

      self.net.step(recvs, blobs, log)

      return recvs, blobs, log

   @runtime
   def step(self):
      '''Broadcasts updated weights to server level
      God optimizer nodes. Performs an Adam step
      once optimizers return a batch of gradients.''' 
      #Update model
      recvs, blobs, log = self.tick()
      
      #Write logs using Quill, checkpoint model
      stats = self.net.quill.stats()
      save  = self.net.saver.log()

      return save, stats, log
