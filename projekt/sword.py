from pdb import set_trace as T
from collections import defaultdict
import numpy as np 
import ray
import projekt

from forge import trinity

from forge.blade.core import realm

from forge.ethyr.io import Stimulus, Action
from forge.ethyr.experience import RolloutManager
from forge.ethyr.torch import Model, optim
from forge.ethyr.torch.param import setParameters

from forge.ethyr.io.io import Output

from copy import deepcopy

from forge.trinity.ascend import Ascend, runtime

#Currently, agents technically run on the same core
#as the environment. This saves 2x cores at small scale
#but will not work with a large number of agents.
#Enable @ray.remote when this becomes an issue.
#@ray.remote
class Sword(Ascend):
   '''Core level Sword API demo

   This core level client node maintains a
   full copy of the model. It runs and computes
   updates for the associated policies of all
   agents.'''

   def __init__(self, trinity, config, idx):
      '''Initializes a model and relevent utilities'''
      super().__init__(None, 0)
      config        = deepcopy(config)
      self.config   = config

      self.net      = projekt.ANN(self.config)
      self.manager  = RolloutManager()

   @runtime
   def step(self, data, recv=None):
      '''Synchronizes weights from upstream; computes
      agent decisions; computes policy updates.'''
      packet, backward = recv
      config           = self.config

      #Sync weights to model
      if packet is not None:
         setParameters(self.net, packet)

      #There may be no obs if NCLIENTS > 1
      if data.obs.n == 0:
         return data, None, None

      #Batch observations and compute forward pass
      self.manager.collectInputs(data)
      self.net(data, self.manager)
  
      if not backward or config.TEST or config.POPOPT:
         return data, None, None

      #Compute backward pass and logs from rollout objects
      rollouts, blobs = self.manager.step()
      optim.backward(rollouts, valWeight=config.VAL_WEIGHT,
         entWeight=config.ENTROPY)#, device=config.DEVICE)
      self.manager.inputs.clear() #Discard partial trajectories

      grads = self.net.grads()
      return data, grads, blobs



