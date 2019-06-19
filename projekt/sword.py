from pdb import set_trace as T
from collections import defaultdict
import numpy as np

import ray
import projekt

from forge import trinity
from forge.trinity.timed import runtime

from forge.blade.core import realm
from forge.blade.io import stimulus
from forge.blade.io.serial import Serial

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
      self.updates      = Serial(self.config)

   @runtime
   def step(self, packet=None):
      '''Synchronizes weights from upstream and
      collects a fixed amount of experience.'''
      self.net.recvUpdate(packet)

      while len(self.updates) < self.config.SYNCUPDATES:
         self.tick()

      return self.updates.finish()

   def tick(self):
      '''Steps the agent and environment

      Processes observations, selects actions, and
      steps the environment to obtain new observations.
      Serializes (obs, action, reward) triplets for
      communication to an upstream optimizer node.'''
 
      #Batch observations and make decisions
      stims = [self.config.dynamic(ob) for ob in self.obs]
      batched = stimulus.Dynamic.batch(stims)
      atnArgs, outputs, values = self.net(batched, obs=self.obs)

      #Update experience buffer
      atns = []
      for ob, stim, atnArg, out, val in zip(
            self.obs, stims, atnArgs, outputs, values):
         env, ent = ob
         entID    = ent.entID
         annID    = ent.annID

         atns.append((entID, atnArg))
         self.updates.serialize(
            self.env, #only used for serialization keys
            env, ent, #raw observation, also for keys 
            stim,     #processed observation (obs)
            out)      #selected actions (action)  

      #Step the environment and all agents at once.
      #The environment handles action priotization etc.
      self.obs, rewards, done, info = super().step(atns)
      self.updates.rewards(rewards) #Injects rewards into the triplet
