from pdb import set_trace as T
from collections import defaultdict
from copy import deepcopy

import numpy as np 
import ray

import projekt
from forge import trinity
from forge.blade.core import realm
from forge.trinity.ascend import Ascend, runtime
from forge.ethyr.experience import RolloutManager
from forge.ethyr.io import Stimulus, Action
from forge.ethyr.io.io import Output
from forge.ethyr.torch import Model, optim
from forge.ethyr.torch.param import setParameters

#@ray.remote
class Sword(Ascend):
   '''Client level infrastructure demo

   This environment server runs a subset of the
   agents associated with a single server and
   computes model updates over collected rollouts

   At small scale, each server is collocated with a
   single client on the same CPU core. For larger
   experiments with multiple clients, decorate this
   class with @ray.remote to enable sharding.'''

   def __init__(self, trinity, config, idx):
      '''Initializes a model and relevent utilities'''
      super().__init__(disciple=None, n=0)
      config        = deepcopy(config)
      self.config   = config

      self.net      = projekt.ANN(self.config)
      self.manager  = RolloutManager()

   @runtime
   def step(self, data, recv=None):
      '''Synchronizes weights from upstream; computes
      agent decisions; computes policy updates.'''
      packet, backward = recv
      grads, blobs     = None, None

      #Sync model weights; batch obs; compute forward pass
      setParameters(self.net, packet)
      self.manager.collectInputs(data)
      self.net(data, self.manager)
  
      #Compute backward pass and logs from full rollouts,
      #discarding any partial trajectories
      if backward and not self.config.TEST:
         rollouts, blobs = self.manager.step()
         optim.backward(rollouts, self.config)
         self.manager.inputs.clear()
         grads = self.net.grads()

      return data, grads, blobs

