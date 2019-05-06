from pdb import set_trace as T
from collections import defaultdict
import numpy as np

import ray, pickle

from forge import trinity
from forge.trinity import Base 
from forge.ethyr.torch.param import setParameters, zeroGrads
from forge.ethyr.torch import optim
from forge.ethyr.rollouts import Rollout
from forge.ethyr.torch import param

from forge.blade.core import realm
from copy import deepcopy

@ray.remote
class Sword(Base.Sword):
   def __init__(self, trin, config, args):
      super().__init__(trin, config, args)
      self.config, self.args = config, args
      self.net = trinity.ANN(config)
      self.updates = []
      ##################################
      #MAKE SURE TO ADD IN ENV SAMPLING#
      ##################################
      self.obs = self.env.reset()

   def collectSteps(self, obs, actions, rewards):
      packets = list(zip(*(obs, actions, rewards)))
      self.updates += packets

   def sync(self):
      updates = self.updates
      self.updates = []
      return pickle.dumps(updates)

   def step(self, packet=None):
      if packet is not None:
         self.net.recvUpdate(packet)
      while len(self.updates) < self.config.SYNCUPDATES:
         self.obs = self.decide(self.obs)
      return self.sync()

   def decide(self, obs):
      atns = []
      for ob in obs:
         env, ent = ob
         entID, annID = ent.entID, ent.annID
         actions, outs, val = self.net(annID, env, ent)
         if actions is not None:
            atns.append((entID, actions))
      obs = deepcopy(obs)
      nxtObs, rewards, done, info = super().step(atns)
      self.collectSteps(obs, atns, rewards)
      return nxtObs
