from pdb import set_trace as T
import numpy as np
import time
import ray
import pickle
from collections import defaultdict

import projekt

from forge.blade.core.realm import Realm
from forge.blade.lib.log import BlobSummary 

from forge.ethyr.io import Stimulus, Action, utils
from forge.ethyr.io import IO

from forge.ethyr.torch import optim
from forge.ethyr.experience import RolloutManager

from forge.trinity.ascend import Ascend, runtime, Log

@ray.remote
class God(Ascend):
   '''Server level infrastructure demo

   This environment server runs a persistent game instance
   that distributes agents observations and collects action
   decisions from a set of client nodes.

   This infrastructure configuration is the direct opposite
   of the typical MPI broadcast/recv paradigm, which operates 
   an optimizer server over a bank of environments. The MPI
   approach does not scale well to massively multiagent
   environments, as it assumes a many-to-one (as opposed to
   a one-to-many) ratio of envs to optimizers.

   OpenAI Rapid style optimization does not fix this
   issue. Rapid solves bandwidth constraints by collocating
   the agent policies with the environment, but it still
   makes the same many-to-one assumption. 

   Note that you could probably get away with either MPI
   or Rapid style optimization on current instances of
   the environment because we only have 128 agents, not
   thousands. However, we built with future scale in
   mind and invested in infrastructure early.''' 

   def __init__(self, trinity, config, idx):
      '''Initializes an environment and logging utilities'''
      super().__init__(trinity.sword, config.NSWORD, trinity, config)
      self.nUpdates, self.grads = 0, []
      self.config, self.idx     = config, idx
      self.nPop, self.ent       = config.NPOP, 0

      self.blobs    = BlobSummary()
      self.env      = Realm(config, idx, self.spawn)

      self.obs, self.rewards, self.dones, _ = self.env.reset()

   def getEnv(self):
      '''Returns the environment. Ray does not allow
      access to remote attributes without a getter'''
      return self.env

   def spawn(self):
      '''Specifies the environment protocol for adding players

      Returns:
         entID (int), popID (int), name (str): 
         unique IDs for the entity and population, 
         as well as a name prefix for the agent 
         (the ID is appended automatically).

      Notes:
         This is useful for population based research,
         as it allows one to specify per-agent or
         per-population policies'''

      pop      =  hash(str(self.ent)) % self.nPop
      self.ent += 1

      return self.ent, pop, 'Neural_'

   def batch(self, nUpdates):
      '''Set backward pass flag and reset update counts
      if the end of the data batch has been reached

      Note: the actual batch size will be smaller than
      specified due to discarded partial trajectories'''
      SERVER_UPDATES = self.config.SERVER_UPDATES
      TEST           = self.config.TEST

      self.backward  =  False
      self.nUpdates  += nUpdates

      if not TEST and self.nUpdates > SERVER_UPDATES:
         self.backward = True
         self.nUpdates = 0

      return self.backward
 
   def distrib(self, weights):
      '''Shards input data across clients using the Ascend async API'''

      def groupFn(entID):
         '''Hash function for mapping entID->popID'''
         return entID % self.config.NSWORD

      #Preprocess obs
      clientData, nUpdates = IO.inputs(
         self.obs, self.rewards, self.dones, 
         groupFn, self.config, serialize=True)

      #Handle possible end of batch
      backward = self.batch(nUpdates)

      #Shard entities across clients
      return super().distrib(clientData, weights, backward, shard=(1, 0, 0))

   def sync(self, rets):
      '''Aggregates output data across shards with the Ascend async API'''
      atnDict, gradList, blobList = None, [], []
      for obs, grads, blobs in super().sync(rets):
         #Process outputs
         atnDict = IO.outputs(obs, atnDict)

         #Collect update
         if self.backward:
            self.grads.append(grads)
            blobList.append(blobs)

      #Aggregate logs
      self.blobs = BlobSummary.merge([self.blobs, *blobList])
      return atnDict

   @runtime
   def step(self, recv):
      '''Sync weights and compute a model update by collecting
      a batch of trajectories from remote clients.'''
      self.grads, self.blobs = [], BlobSummary()
      while len(self.grads) == 0:
         self.tick(recv)
         recv = None

      #Aggregate updates and logs
      grads = np.mean(self.grads, 0)
      log   = Log.summary([
                  self.discipleLogs(), 
                  self.env.logs()])

      return grads, self.blobs, log

   #Note: IO is currently slow relative to the
   #forward pass. The backward pass is fast relative to
   #the forward pass but slow relative to IO.
   def tick(self, recv=None):
      '''Simulate a single server tick and all remote clients.

      The optional data packet specifies a new model parameter vector
      '''
      #Make decisions
      actions = super().step(recv)

      #Step the environment and all agents at once.
      #The environment handles action priotization etc.
      self.obs, self.rewards, self.dones, _ = self.env.step(actions)
