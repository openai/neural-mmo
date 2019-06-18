'''Module level sword'''

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
from forge.ethyr.torch import optim
from forge.ethyr.torch import param

from forge.blade.core import realm
from forge.blade.io import stimulus
from forge.blade.io.serial import Serial

from copy import deepcopy

@ray.remote
class Sword(Base.Sword):
   '''Core level agent module'''
   def __init__(self, trin, config, args, idx):
      '''Core level agent initializer

      Args:
         trin: trinity

      '''
      super().__init__(trin, config, args, idx)
      config        = deepcopy(config)
      config.DEVICE = 'cpu:0'

      self.config   = config
      self.args     = args

      self.net = trinity.ANN(config)
      self.obs = self.env.reset()
      self.ent = 0

      self.updates = Serial(self.config)
      self.first = True

   def spawn(self):
      ent = self.ent
      pop = hash(str(ent)) % self.config.NPOP
      self.ent += 1
      return ent, pop, 'Neural_'
 
   @runtime
   def step(self, packet=None):
      '''Accept upstream packet and return updates'''
      if packet is not None:
         self.net.recvUpdate(packet)

      while len(self.updates) < self.config.SYNCUPDATES:
         self._step()

      updates = self.updates.finish()
      return updates

   def _step(self):
      atns     = self.decide(self.obs)
      self.obs = self.stepEnv(atns)

   def stepEnv(self, atns):
      nxtObs, rewards, done, info = super().step(atns)
      self.updates.rewards(rewards)
      return nxtObs

   def decide(self, obs):
      atns = []
      if len(obs) == 0:
         return atns

      #stims = 
      obbys = [self.config.dynamic(ob) for ob in obs]
      #Is this one needed?
      #obbys = deepcopy(stims)
      stims = stimulus.Dynamic.batch(obbys)
      if self.first:
         self.first = False
         iden = (self.env.worldIdx, self.env.tick)
         serial = stims['Entity'][0][0][0].serial
         #print('Key: ', iden + serial)
         #print(stims)
         #print()

      #Make decisions
      atnArgs, outputs, values = self.net(stims, obs=obs)

      #Update experience buffer
      for obs, obby, atnArg, out, val in zip(
            obs, obbys, atnArgs, outputs, values):
         env, ent = obs
         entID, annID = ent.entID, ent.annID
         atns.append((entID, atnArg))

         iden = (self.env.worldIdx, self.env.tick)
         self.updates.serialize(env, ent, obby, out, iden)

      return atns
