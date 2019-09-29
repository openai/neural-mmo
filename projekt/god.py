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

import torch

from forge.trinity.ascend import Ascend, runtime, Log

#Figure out why the fuck this isn't spliting shard time on sync
@ray.remote(num_gpus=1/6)
class God(Ascend):
   '''Server level God API demo

   This environment server runs a persistent game instance
   It distributes agents observations and collects action
   decisions from a set of client nodes. This is the direct
   opposite of the typical MPI broadcast/recv paradigm, which
   operates an optimizer server over a bank of environments.

   Notably, OpenAI Rapid style optimization does not scale
   well to to this setting, as it assumes environment and
   agent collocation. This quickly becomes impossible when
   the number of agents becomes large. While this may not
   be a problem for small scale versions of this demo, it
   is built with future scale in mind.'''

   def __init__(self, trinity, config, idx):
      '''Initializes a model and relevent utilities'''
      super().__init__(trinity.sword, config.NSWORD, trinity, config)
      self.config, self.idx = config, idx
      self.nPop = config.NPOP
      self.ent  = 0

      self.env  = Realm(config, idx, self.spawn)
      self.obs, self.rewards, self.dones, _ = self.env.reset()

      self.grads, self.blobs = [], BlobSummary()

   def getEnv(self):
      '''Returns the environment. Ray does not allow
      access to remote attributes without a getter'''
      return self.env

   def getEnvLogs(self):
      '''Returns the environment logs. Ray does not allow
      access to remote attributes without a getter'''
      return self.env.logs()

   def getDisciples(self):
      '''Returns client instances. Ray does not allow
      access to remote attributes without a getter'''
      return self.disciples

   def getSwordLogs(self):
      '''Returns client logs. Ray does not allow
      access to remote attributes without a getter'''
      return [e.logs.remote() for e in self.disciples]

   def spawn(self):
      '''Specifies how the environment adds players

      Returns:
         entID (int), popID (int), name (str): 
         unique IDs for the entity and population, 
         as well as a name prefix for the agent 
         (the ID is appended automatically).

      Notes:
         This is useful for population based research,
         as it allows one to specify per-agent or
         per-population policies'''

      pop = hash(str(self.ent)) % self.nPop
      self.ent += 1
      return self.ent, pop, 'Neural_'

   def distrib(self, *args):
      '''Shards observation data across clients using the Trinity async API'''
      obs, recv = args
      N = self.config.NSWORD
      clientData = [[] for _ in range(N)]
      for ob in obs:
         clientData[ob.entID % N].append(ob)
      return super().distrib(clientData, recv, shard=(1, 0))

   def sync(self, rets):
      '''Aggregates actions/updates/logs from shards using the Trinity async API'''
      rets = super().sync(rets)

      atnDict, gradList, blobList = {}, [], []#, logList = {}, [], [], []
      for atns, grads, blobs in rets:
         atnDict.update(atns)

         #Gradients only present when a particular
         #client has computed a batch
         #logList.append(logs)

         if grads is not None:
            gradList.append(grads)

         if blobs is not None:
            blobList.append(blobs)

      #Aggregate gradients/logs
      self.grads += gradList
      self.blobs = BlobSummary.merge([self.blobs, *blobList])
      #self.log   = Log.summary(logList)

      return atnDict

   @runtime
   def step(self, recv):
      '''Broadcasts updated weights to the core level clients.
      Collects gradient updates for upstream optimizer.'''
      #self.resetLogs()
      self.grads, self.blobs = [], BlobSummary()
      while self.blobs.nUpdates < self.config.SERVER_UPDATES:
         self.tick(recv)
         recv = None

      grads = np.mean(self.grads, 0)

      swordLog = self.discipleLogs()
      realmLog = self.env.logs()
      log = Log.summary([swordLog, realmLog])
      return grads, self.blobs, log

   #Note: IO is currently slow relative to the
   #forward pass. The backward pass is fast relative to
   #the forward pass but slow relative to IO.
   def tick(self, recv=None):
      '''Environment step Thunk. Optionally takes a data packet.

      In this case, the data packet arg is used to specify
      model updates in the form of a new parameter vector'''

      #Preprocess obs
      obs, rawAtns = IO.inputs(
         self.obs, self.rewards, self.dones, 
         self.config, serialize=True)

      #Make decisions
      atns = super().step(obs, recv)

      #Postprocess outputs
      actions = IO.outputs(obs, rawAtns, atns)

      #Step the environment and all agents at once.
      #The environment handles action priotization etc.
      self.obs, self.rewards, self.dones, _ = self.env.step(actions)
