from pdb import set_trace as T
from collections import defaultdict
import numpy as np

import ray
import projekt

from forge import trinity
from forge.trinity.timed import runtime

from forge.blade.core import realm
from forge.blade.io import stimulus

from forge.ethyr.buffer import RolloutManager

from copy import deepcopy

@ray.remote
class Sword(trinity.Sword):
   '''Core level Sword API demo

   This core level rollout worker node runs
   a copy of the environment and all associated
   agents. Multiple Swords return observations, 
   actions, and rewards to each server level 
   optimizer node.'''

   def __init__(self, trin, config, args, idx):
      '''Initializes a model, env, and relevent utilities'''

      super().__init__(trin, config, args, idx)
      config        = deepcopy(config)
      config.DEVICE = 'cpu:0'

      self.config   = config
      self.args     = args
      self.ent      = 0

      self.net          = projekt.ANN(config)
      self.obs, _, _, _ = self.env.reset()

   @runtime
   def step(self, packet=None):
      '''Synchronizes weights from upstream and
      collects a fixed amount of experience.'''
      self.net.recvUpdate(packet)

      self.manager = RolloutManager()
      while self.manager.nUpdates < self.config.SYNCUPDATES:
         self.tick()

      return self.manager.send()

   def tick(self):
      '''Steps the agent and environment

      Processes observations, selects actions, and
      steps the environment to obtain new observations.
      Serializes (obs, action, reward) triplets for
      communication to an upstream optimizer node.'''
 
      #Batch observations and make decisions
      stims   = [self.config.dynamic(ob) for ob in self.obs]
      batched = stimulus.Dynamic.batch(stims)

      #Run the policy
      actions, outs, _ = self.net(batched, obs=self.obs)

      #Step the environment and all agents at once.
      #The environment handles action priotization etc.
      actions = dict(((o[1].entID, a) for o, a in zip(self.obs, actions)))
      nxtObs, rewards, dones, info = super().step(actions)

      #Update the experience buffer
      #The envrionment is used to generate serialization keys
      self.manager.collect(
         self.env, self.obs, stims, outs, rewards, dones)
      self.obs = nxtObs

