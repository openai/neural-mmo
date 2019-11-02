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
      self.ent      = 0

      self.keys = set()

      self.net = projekt.ANN(self.config)
      #self.ents = {}
      self.manager = RolloutManager()

   @runtime
   def step(self, obs, recv=None):
      '''Synchronizes weights from upstream; computes
      agent decisions; computes policy updates.
      
      A few bug notes:
         1. It appears pytorch errors in .backward when batching
         data. This is because the graph is retained over all
         trajectories in the batch, even though only some are
         finished.
         
         2. Currently specifying retain_graph. This should not be
         required with batch size 1, even with the above bug.
      '''
      packet, backward = recv

      #Sync weights
      if packet is not None:
         setParameters(self.net, packet)

      config  = self.config
      actions = {}

      if len(obs) == 0:
         return actions, None, None

      #Batch observations
      inputs, data, dataLookup = self.manager.collectInputs(obs)

      #Compute forward pass
      keys, atns, atnsIdx, vals = self.net(inputs, data, dataLookup)

      #Clear .backward buffers during test
      if self.config.TEST or self.config.POPOPT:
         #atns are detached in torch/io/action
         atnsIdx = atnsIdx.detach()
         vals    = vals.detach()

      #Collect output actions and values for .backward
      for key, atn, atnIdx, val in zip(keys, atns, atnsIdx, vals):
         out = Output(key, atn, atnIdx, val)
         actions.update(out.action)
         self.manager.collectOutputs([out])
         
      #Compute backward pass and logs from rollout objects
      #if self.manager.nUpdates >= config.CLIENT_UPDATES:
      if backward:
         rollouts, blobs = self.manager.step()

         if config.TEST or config.POPOPT:
            return actions, None, blobs

         optim.backward(rollouts, valWeight=config.VAL_WEIGHT,
            entWeight=config.ENTROPY)#, device=config.DEVICE)

         grads = self.net.grads()
         return actions, grads, blobs

      return actions, None, None


