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
from forge.ethyr.torch.serial import Serial, Stim
from forge.ethyr.torch import optim
from forge.ethyr.torch import param

from forge.blade.core import realm
from forge.blade.io import stimulus

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


   @runtime
   def step(self, packet=None):
      if packet is not None:
         self.net.recvUpdate(packet)

      self.updates = Serial(self.config)
      while len(self.updates) < self.config.SYNCUPDATES:
         self.obs = self.decide(self.obs)

      return self.updates.finish()

   def decide(self, obs):
      atns = []
      #Could potentially parallelize the inputs
      if len(obs) > 0:
         obbys = [self.config.dynamic(ob, flat=True) for ob in obs]
         stims = stimulus.Dynamic.batch(obbys)

         #Make decisions
         actions, outList, vals = self.net(stims, obs=obs)
         for obs, obby, action, outs, val in zip(obs, obbys, actions, outList, vals):
            env, ent = obs
            entID, annID = ent.entID, ent.annID
            atns.append((entID, action))
            #Update experience buffer
            iden = (self.env.worldIdx, self.env.tick)
            self.updates.serialize(env, ent, obby, outs, iden)

     

      '''
      for ob in obs:
         env, ent = ob
         stim = self.config.dynamic(ob, flat=True)
         obby = stim

         #Batch inputs
         entID, annID = ent.entID, ent.annID
         stim  = stimulus.Dynamic.batch([stim])
         
         #Make decisions
         actions, outs, val = self.net(stim, env, ent)
         atns.append((entID, actions))

         #Update experience buffer
         iden = (self.env.worldIdx, self.env.tick)
         self.updates.serialize(env, ent, obby, outs, iden)
      '''

      #Step environment
      nxtObs, rewards, done, info = super().step(atns)
      self.updates.rewards(rewards)
      return nxtObs

   def spawn(self):
      ent = self.ent
      pop = hash(str(ent)) % self.config.NPOP
      self.ent += 1
      return ent, pop, 'Neural_'
      
