from pdb import set_trace as T

import numpy as np
import ray

from collections import defaultdict

from forge.blade import core
from forge.blade.lib.log import BlobSummary 

from forge.trinity.ascend import Ascend, runtime, Log
from forge.blade import IO

from forge.ethyr.torch.param import setParameters

import projekt

class Realm(core.Realm):
   '''Example environment overrides'''
   def spawn(self):
      '''Example override of the spawn function'''
      return super().spawn()

      #Example override
      ent    = self.desciples[entID]
      packets[entID].reward = 0.05 * min(
          ent.resources.health.val,
          ent.resources.water.val,
          ent.resources.food.val)

   def reward(self, ent):
      '''Example override of the reward function'''
      return super().reward(ent)

@ray.remote(num_gpus=0)
class God(Ascend):
   '''Server level infrastructure layer 

   This environment server runs a persistent game instance that distributes 
   agents observations and collects action decisions from client nodes.

   This infrastructure configuration is the direct opposite of the typical 
   MPI broadcast/recv paradigm, which operates an optimizer server over a 
   bank of environments. The MPI approach does not scale well to massively
   multiagent environments, as it assumes a many-to-one (as opposed to a 
   one-to-many) ratio of envs to optimizers.

   OpenAI Rapid style optimization does not fix this issue. Rapid solves 
   bandwidth constraints by collocating the agent policies with the 
   environment, but it still makes the same many-to-one assumption. 

   Note that you could probably get away with either MPI or Rapid style
   optimization on current instances of the environment because we only have
   128 agents, not thousands. However, we built with future scale in mind
   and invested in infrastructure early.''' 

   def __init__(self, trinity, config, idx):
      '''Initializes an environment and logging utilities

      Args:
         trinity : A Trinity object as shown in __main__
         config  : A Config object as shown in __main__
         idx     : Hardware index used to specify the game map
      '''
      super().__init__(trinity.sword, config.NSWORD, trinity, config)
      self.nPop, self.ent       = config.NPOP, 0
      self.config, self.idx     = config, idx
      self.nUpdates, self.grads = 0, []

      self.env      = Realm(config, idx)
      self.blobs    = BlobSummary()

      self.obs, self.rewards, self.dones, _ = self.env.reset()

   def getEnv(self):
      '''Ray does not allow direct access to remote attributes

      Returns:
         env: A copy of the environment (returns a reference in local mode)
      '''
      return self.env

   def clientHash(self, entID):
      '''Hash function for mapping entID->client

      Returns:
         clientID: A client membership ID
      '''
      return entID % self.config.NSWORD

   def batch(self, nUpdates):
      '''Set backward pass flag and reset update counts
      if the end of the data batch has been reached

      Args:
         nUpdates: The number of agent steps collected since the last call

      Returns:
         backward: (bool) Whether of not a backward pass should be performed

      Note: the actual batch size will be smaller than set in the
      configuration file due to discarded partial trajectories'''
      SERVER_UPDATES = self.config.SERVER_UPDATES
      TEST           = self.config.TEST

      self.backward  =  False
      self.nUpdates  += nUpdates

      if not TEST and self.nUpdates > SERVER_UPDATES:
         self.backward = True
         self.nUpdates = 0

      return self.backward

   def values(self, weights):
      print('Computing value map...')
      obs, rewards, dones, _ = self.env.getValStim()
      packet, _ = IO.inputs(obs, rewards, dones, self.config)
      net = projekt.Policy(self.config)
      setParameters(net, weights)
      net(packet, None)
      atnDict, values, attn = IO.outputs(packet)
      
      R, C = self.env.world.env.tiles.shape
      globalValues = np.zeros((R, C))
      for env, ent in obs:
         r, c = ent.base.pos
         globalValues[r, c] = values[ent.entID]
         #dist = np.sqrt((r/40-1)**2 + (c/40-1)**2)
         #globalValues[r, c] = dist

      self.env.setGlobalValues(globalValues)
      print('Value map computed')
 
   def distribute(self, weights):
      '''Shards input data across clients using the Ascend async API

      Args:
         weights: A parameter vector to replace the current model weights

      Returns:
         rets: List of (data, gradients, blobs) tuples from the clients
      '''

      #Preprocess obs
      clientData, nUpdates = IO.inputs(
         self.obs, self.rewards, self.dones, 
         self.config, self.clientHash)

      #Handle possible end of batch
      backward = self.batch(nUpdates)

      #Shard entities across clients
      return super().distribute(clientData, weights, backward, shard=(1, 0, 0))

   def synchronize(self, rets):
      '''Aggregates output data across shards with the Ascend async API

      Args:
         rets: List of (data, gradients, blobs) tuples from self.distrib()

      Returns:
         atnDict: Dictionary of actions to be submitted to the environment
      '''
      atnDict, gradList, blobList = None, [], []
      values, attn = None, None
      for obs, grads, blobs in super().synchronize(rets):
         #Process outputs
         atnDict, values, attn = IO.outputs(obs, atnDict, values, attn)

         #Collect update
         if self.backward:
            self.grads.append(grads)
            self.blobs.add([blobs])

      return atnDict, values, attn

   @runtime
   def step(self, recv):
      '''Sync weights and compute a model update by collecting
      a batch of trajectories from remote clients.

      Args:
         recv: Upstream data from the cluster (in this case, a param vector)

      Returns:
         grads   : A vector of gradients aggregated across clients
         summary : A BlobSummary object logging agent statistics
         log     : Logging object for infrastructure timings
      '''
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

   def tick(self, recv=None):
      '''Simulate a single server tick and all remote clients.
      The optional data packet specifies a new model parameter vector

      Args:
         recv: Upstream data from the cluster (in this case, a param vector)
      '''
      #Make decisions
      actions, values, attn = super().step(recv)

      #Step the environment and all agents at once.
      #The environment handles action priotization etc.
      self.obs, self.rewards, self.dones, _ = self.env.step(actions, values, attn)
