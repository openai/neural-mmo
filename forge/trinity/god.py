from pdb import set_trace as T
import numpy as np

import time
import ray
import pickle
from collections import defaultdict

from forge import trinity
from forge.trinity import Base

from forge.ethyr.torch import optim
from forge.ethyr.rollouts import Rollout

class ExperienceBuffer:
   def __init__(self):
      self.data = defaultdict(list)
      self.rollouts = {}
      self.nRollouts = 0

   def __len__(self):
      return self.nRollouts

   def collect(self, packets):
      for sword, packetList in enumerate(packets):
         for packet in packetList:
            obs, actions, reward = packet
            env, ent = obs

            entID, time = ent.entID, ent.timeAlive
            key = (sword, entID)

            assert len(self.data[key]) == time.val
            self.data[key].append(packet)

            if reward == -1:
               self.rollouts[key] = self.data[key]
               del self.data[key]
               self.nRollouts += 1

   def gather(self):
      rollouts = self.rollouts
      self.nRollouts = 0
      self.rollouts = {}
      return rollouts

@ray.remote(num_gpus=1)
class God(Base.God):
   def __init__(self, trin, config, args):
      super().__init__(trin, config, args)
      self.config, self.args = config, args
      self.device = 'cuda:0'
      self.net = trinity.ANN(config, self.device).to(self.device)

      self.replay = ExperienceBuffer()
      self.blobs  = []

   def collectStep(self, entID, atnArgs, val, reward):
      if self.config.TEST:
          return
      self.updates[entID].step(atnArgs, val, reward)
      self.updates[entID].feather.scrawl(
            env, ent, val, reward)

   def forward(self):
      data = self.replay.gather()
      for key, traj in data.items():
         rollout = Rollout()
         for packet in traj:
            obs, actions, reward = packet
            env, ent = obs
            actions, outs, val = self.net(ent.annID, env, ent)
            rollout.step(outs, val, reward)
            rollout.feather.scrawl(env, ent, val, reward)
         rollout.finish()
         self.backward({'rollout': rollout})

   def backward(self, rollouts):
      reward, val, pg, valLoss, entropy = optim.backward(
            rollouts, valWeight=0.25,
            entWeight=self.config.ENTROPY, device=self.device)

      self.blobs += [r.feather.blob for r in rollouts.values()]
      self.rollouts = {}
      self.nGrads = 0
      self.networksUsed = set()

   def send(self):
      grads = self.net.grads()
      blobs = self.blobs
      self.blobs = []
      return grads, blobs

   def rollout(self, recv):
      while len(self.replay) < self.config.NROLLOUTS:
         packets = super().step(recv)
         self.replay.collect(packets)
         recv = None
 
   def step(self, recv):
      self.net.recvUpdate(recv)

      sword = time.time()
      self.rollout(recv)
      sword = time.time() - sword

      god   = time.time()
      self.forward()
      god   = time.time() - god

      print('God: ', god, ', Swords: ', sword)
      return self.send()

     
      
      

      
      
