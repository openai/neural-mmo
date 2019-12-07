from pdb import set_trace as T

from collections import defaultdict

from forge.ethyr.torch import Model
from forge.blade.lib.log import Quill, BlobSummary
from forge.trinity.ascend import Ascend, runtime

import projekt

class Pantheon(Ascend):
   '''Cluster level infrastructure layer

   This module aggregates gradients across all server level 
   environments and updates model weights using Adam.

   It also demonstrates logging and snapshotting functionality 
   through the Quill and Model libraries, respectively.'''

   def __init__(self, trinity, config, idx):
      '''Initializes a copy of the model, which keeps
      track of the weights for the optimizer.

      Args:
         trinity : A Trinity object as shown in __main__
         config  : A Config object as shown in __main__
         idx     : Unused hardware index
      '''
      super().__init__(trinity.god, config.NGOD, trinity, config)
      self.log    = defaultdict(list)
      self.config = config

      self.net = Model(projekt.ANN, config)
      self.net.printParams()

   @runtime
   def step(self):
      '''Broadcasts updated weights to server level God optimizer nodes.
      Performs an Adam step once optimizers return a batch of gradients.

      Returns:
         perf  : Log message describing agent performance
         stats : Log message describing data collected
         log   : Dictionary of logs containing infrastructure usage data
      ''' 
      #Aggregate Blob logs as a BlobSummary
      recvs             = super().step(self.net.weights)
      recvs, blobs, log = list(zip(*recvs))
      blobs             = BlobSummary.merge(blobs)

      #Update/checkpoint model and write logs
      self.net.step(recvs, blobs, log)
      stats = self.net.quill.stats()
      perf  = self.net.saver.log()

      return perf, stats, log
