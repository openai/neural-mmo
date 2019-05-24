from pdb import set_trace as T
from collections import defaultdict
import numpy as np

import sys
import ray
import pickle

from forge import trinity
from forge.trinity import Base 
from forge.trinity.timed import runtime

from forge.ethyr.rollouts import Rollout

from forge.ethyr.torch.param import setParameters, zeroGrads
from forge.ethyr.torch.serial import Serial
from forge.ethyr.torch import optim
from forge.ethyr.torch import param

from forge.blade.core import realm
from copy import deepcopy

@ray.remote
class Sword(Base.Sword):
   def __init__(self, trin, config, args, idx):
      super().__init__(trin, config, args, idx)
      self.config, self.args = config, args
      self.updates = Serial(self.config)
      self.net = trinity.ANN(config)
      ##################################
      #MAKE SURE TO ADD IN ENV SAMPLING#
      ##################################
      self.obs = self.env.reset()

      self.nPop = config.NPOP
      self.ent = 0

   def collectUpdates(self):
      updates = self.updates.finish()
      return updates

   @runtime
   def step(self, packet=None):
      if packet is not None:
         self.net.recvUpdate(packet)
      while len(self.updates) < self.config.SYNCUPDATES:
         self.obs = self.decide(self.obs)
      return self.collectUpdates()

   def decide(self, obs):
      atns = []
      for ob in obs:
         env, ent = ob
         stim = self.config.dynamic(ob, flat=True)

         iden = (self.env.worldIdx, self.env.tick)
         entID, annID = ent.entID, ent.annID
         actions, outs, val = self.net(stim, env, ent)

         if actions is not None:
            atns.append((entID, actions))

         self.updates.serialize(env, ent, stim, outs, iden)

      obs = deepcopy(obs)
      nxtObs, rewards, done, info = super().step(atns)
      self.updates.rewards(rewards)
      return nxtObs

   def spawn(self):
      ent = self.ent
      pop = hash(str(ent)) % self.nPop
      self.ent += 1
      return ent, pop, 'Neural_'
      
