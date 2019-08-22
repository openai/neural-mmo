from pdb import set_trace as T
import numpy as np 
import time
import ray
import pickle
from collections import defaultdict

import projekt


from forge import trinity
from forge.blade.core.realm import Realm
from forge.blade.lib.log import BlobLogs
from forge.trinity.timed import runtime

from forge.ethyr.io import Stimulus, Action, utils
from forge.ethyr.io import IO

from forge.ethyr.torch import optim
from forge.ethyr.experience import RolloutManager

import torch

@ray.remote
class God(trinity.God):
   '''Server level God API demo

   This server level optimizer node aggregates experience
   across all core level rollout worker nodes. It
   uses the aggregated experience compute gradients.

   This is effectively a lightweight variant of the
   Rapid computation model, with the potential notable
   difference that we also recompute the forward pass
   from small observation buffers rather than 
   communicating large activation tensors.

   This demo builds up the ExperienceBuffer utility, 
   which handles rollout batching.'''

   def __init__(self, trin, config, args, idx):
      '''Initializes a model and relevent utilities'''
      super().__init__(trin, config, args, idx)
      self.config, self.args = config, args
      self.nPop = config.NPOP
      self.ent  = 0

      self.env  = Realm(config, args, idx, self.spawn)
      self.obs, self.rewards, self.dones, _ = self.env.reset()

      self.grads, self.logs = [], BlobLogs()

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

   def getEnv(self):
      '''Returns the environment. Ray does not allow
      access to remote attributes without a getter'''
      return self.env

   def distrib(self, *args):
      obs, recv = args
      N = self.config.NSWORD
      clientData = [[] for _ in range(N)]
      for idx, ob in enumerate(obs):
         clientData[idx % N].append(ob)
      return super().distrib(obs, recv)

   def sync(self, rets):
      rets = super().sync(rets)

      atnDict, gradList, logList = {}, [], []
      for atns, grads, logs in rets:
         atnDict.update(atns)

         if grads is not None:
            gradList.append(grads)

         if logs is not None:
            logList.append(logs)

      self.grads += gradList
      self.logs  = BlobLogs.merge([self.logs, *logList])

      return atnDict

   @runtime
   def step(self, recv):
      '''Broadcasts updated weights to the core level
      Sword rollout workers. Runs rollout workers'''
      self.grads, self.logs = [], BlobLogs()
      while self.logs.nUpdates < self.config.OPTIMUPDATES:
         self.tick(recv)
         recv = None

      grads = np.mean(self.grads, 0)
      grads = grads.tolist()
      return grads, self.logs

   def tick(self, recv=None):
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
