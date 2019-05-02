from pdb import set_trace as T
from collections import defaultdict
import numpy as np

from forge import trinity
from forge.trinity import Base 
from forge.ethyr.torch.param import setParameters, zeroGrads
from forge.ethyr.torch import optim
from forge.ethyr.rollouts import Rollout
from forge.ethyr.torch import param

from forge.blade.core import realm
from copy import deepcopy

class Sword(Base.Sword):
   def __init__(self, god, config, args):
      self.god, self.config, self.args = god, config, args
      self.net = trinity.ANN(config)
      self.updates = []
      ##################################
      #MAKE SURE TO ADD IN ENV SAMPLING#
      ##################################
      self.env = realm.VecEnvRealm(config, args, idx=0) 
      self.obs = self.env.reset()

   def collectStep(self, env, ent, actions):
      packet = deepcopy((env, ent, actions))
      self.updates.append(packet)

   def sync(self):
      updates = self.updates
      self.updates = []
      return updates

   def run(self, packet=None):
      if packet is not None:
         self.net.recvUpdate(packet)
      while len(self.updates) < self.config.SYNCUPDATES:
         self.obs = self.step(self.obs)
      return self.sync()

   def step(self, obs):
      atns = []
      for ob in obs:
         entID, actions = self.decide(*ob)
         if actions is not None:
            atns.append((entID, actions))
      obs, rewards, done, info = self.env.step(atns)
      return obs

   def decide(self, env, ent):
      entID, annID = ent.entID, ent.annID
      #Need to copy ent and obs
      if ent.alive and not ent.kill:
         actions, outs, val = self.net(annID, env, ent)
      else:
         actions = None
      self.collectStep(env, ent, actions)
      return entID, actions
