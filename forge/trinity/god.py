from pdb import set_trace as T
import numpy as np

import ray
from collections import defaultdict

from forge import trinity
from forge.trinity import Base

from forge.ethyr.torch import optim
from forge.ethyr.rollouts import Rollout

#@ray.remote(num_gpus=1)
class AsyncQueue:
   def __init__(self):
      self.data = []

   def put(self, data):
      self.data += data

   def pop(self, n):
      dat = self.data.pop(n)
      return dat

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
            env, ent, actions = packet
            entID, time = ent.entID, ent.timeAlive
            key = (sword, entID)

            if ent.alive and not ent.kill:
               assert len(self.data[key]) == time.val
               self.data[key].append(packet)
            else:
               self.rollouts[key] = self.data[key]
               del self.data[key]
               self.nRollouts += 1

   def gather(self):
      self.nRollouts = 0
      return self.rollouts


#@ray.remote(num_gpus=1)
class God(Base.God):
   def __init__(self, trin, config, args):
      super().__init__(trin, config, args)
      self.config, self.args = config, args
      self.net = trinity.ANN(config)

      self.replay = ExperienceBuffer()
      self.blobs  = []
      #self.rollouts = defaultdict(Rollout)

   def collectStep(self, entID, atnArgs, val, reward):
      if self.config.TEST:
          return
      self.updates[entID].step(atnArgs, val, reward)
      self.updates[entID].feather.scrawl(
            env, ent, val, reward)

   def collectRollout(self, entID, ent):
      assert entID not in self.rollouts
      rollout = self.updates[entID]
      rollout.finish()
      self.nGrads += rollout.lifespan
      self.rollouts[entID] = rollout
      del self.updates[entID]

      # assert ent.annID == (hash(entID) % self.nANN)
      self.networksUsed.add(ent.annID)


   def forward(self):
      data = self.replay.gather()
      for key, traj in data.items():
         rollout = Rollout()
         for packet in traj:
            env, ent, actions = packet
            actions, outs, val = self.net(ent.annID, env, ent)
            rollout.step(outs, val, 1)
         rollout.finish()
         self.backward({'rollout': rollout})

   def backward(self, rollouts):
      reward, val, pg, valLoss, entropy = optim.backward(
            rollouts, valWeight=0.25,
            entWeight=self.config.ENTROPY)

      self.blobs += [r.feather.blob for r in rollouts.values()]
      self.rollouts = {}
      self.nGrads = 0
      self.networksUsed = set()

   def send(self):
      grads = self.net.grads()
      blobs = self.blobs
      self.blobs = []
      return grads, blobs

   def run(self, recv):
      self.net.recvUpdate(recv)
      while len(self.replay) < self.config.NROLLOUTS:
         packets = self.step(recv)
         self.replay.collect(packets)
         recv = None
      self.forward()
      return self.send()

     
      
      

      
      
