from pdb import set_trace as T
from copy import deepcopy

import ray

from forge.trinity.ascend import Ascend, runtime

from forge.ethyr.experience import RolloutManager
from forge.ethyr.torch.param import setParameters
from forge.ethyr.torch import optim

import projekt

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
      '''Initializes a model and relevent utilities
                                                                              
      Args:                                                                   
         trinity : A Trinity object as shown in __main__                      
         config  : A Config object as shown in __main__                       
         idx     : Unused hardware index                                      
      '''
      super().__init__(disciple=None, n=0)
      config        = deepcopy(config)
      self.config   = config

      self.net      = projekt.Policy(config)
      self.manager  = RolloutManager(config)

   @runtime
   def step(self, packet, weights, backward):
      '''Synchronizes weights from upstream; computes
      agent decisions; computes policy updates.
                                                                              
      Args:                                                                   
         packet   : An IO object specifying observations
         weights  : An optional parameter vector to replace model weights
         backward : (bool) Whether of not a backward pass should be performed  

      Returns:                                                                   
         data    : The same IO object populated with action decisions
         grads   : A vector of gradients aggregated across trajectories
         summary : A BlobSummary object logging agent statistics
      '''   
      grads, blobs = None, None

      #Sync model weights; batch obs; compute forward pass
      setParameters(self.net, weights)
      self.manager.collectInputs(packet)
      self.net(packet, self.manager)
  
      #Compute backward pass and logs from full rollouts,
      #discarding any partial trajectories
      if backward and not self.config.TEST:
         rollouts, blobs = self.manager.step()
         optim.backward(rollouts, self.config)
         self.manager.inputs.clear()
         grads = self.net.grads()

      return packet, grads, blobs

