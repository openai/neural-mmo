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
   def __init__(self, trin, config, args, idx):
      super().__init__(trin, config, args, idx)
      config        = deepcopy(config)
      config.DEVICE = 'cpu:0'

      self.config   = config
      self.args     = args

      self.net = trinity.ANN(config)
      self.obs = self.env.reset()
      self.ent = 0

      self.updates = Serial(self.config)

   def spawn(self):
      ent = self.ent
      pop = hash(str(ent)) % self.config.NPOP
      self.ent += 1
      return ent, pop, 'Neural_'
 
   @runtime
   def step(self, packet=None):
      if packet is not None:
         self.net.recvUpdate(packet)

      while len(self.updates) < self.config.SYNCUPDATES:
         self._step()

      return self.updates.finish()

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

      obbys = [self.config.dynamic(ob) for ob in obs]
      stims = stimulus.Dynamic.batch(obbys)

      #Make decisions
      actions, outList, vals = self.net(stims, obs=obs)

      #Update experience buffer
      for obs, obby, action, outs, val in zip(
            obs, obbys, actions, outList, vals):

         env, ent = obs
         entID, annID = ent.entID, ent.annID
         atns.append((entID, action))

         iden = (self.env.worldIdx, self.env.tick)
         self.updates.serialize(env, ent, obby, outs, iden)

      return atns
