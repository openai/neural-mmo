from pdb import set_trace as T
import numpy as np

import time
import ray
import pickle
from collections import defaultdict

from forge import trinity
from forge.trinity import Base
from forge.trinity.timed import runtime

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

   def flat(self, rollouts):
      keys, obs, actions, rewards = [], [], [], []
      for key, rollout in rollouts:
         for val in rollout:
            ob, atns, reward = val
            keys.append(key)
            obs.append(ob)
            actions.append(atns)
            rewards.append(reward)
      return keys, obs, actions, rewards

   def batch(self, sz):
      data = []
      rollouts = list(self.gather().items())
      print('Num rollouts: ', len(rollouts))
      while len(rollouts) > 0:
         dat = rollouts[:sz]
         rollouts = rollouts[sz:]
         dat = self.flat(dat)
         data.append(dat)
      return data

@ray.remote(num_gpus=1)
class God(Base.God):
   def __init__(self, trin, config, args):
      super().__init__(trin, config, args)
      self.config, self.args = config, args
      self.device = 'cuda:0'
      self.net = trinity.ANN(config, self.device).to(self.device)

      self.replay = ExperienceBuffer()
      self.blobs  = []

   #Something is fucked here. Step through logic
   def forward(self, keys, obs, actions, rewards):
      rollouts = defaultdict(Rollout)
      _, outs, vals = self.net(obs)

      rets = zip(keys, obs, actions, rewards, outs, vals)
      for key, ob, action, reward, out, val in rets:
         env, ent = ob
         rollout = rollouts[key]
         rollout.step(out, val, reward)
         rollout.feather.scrawl(env, ent, val, reward)

      for key, rollout in rollouts.items():
         rollout.finish()

      return rollouts

   def backward(self, rollouts):
      reward, val, pg, valLoss, entropy = optim.backward(
            rollouts, valWeight=0.25,
            entWeight=self.config.ENTROPY, device=self.device)

      self.blobs += [r.feather.blob for r in rollouts.values()]

   def processReplay(self):
      batches = self.replay.batch(self.config.BATCH)
      for batch in batches:
         rollouts = self.forward(*batch)
         self.backward(rollouts)

   def rollout(self, packets, recv=None):
      packets = super().distrib(recv)
      self.processReplay()
      packets = super().sync(packets)

      self.replay.collect(packets)
      self.nRollouts += len(self.replay)
      return packets

   def collectGrads(self, recv):
      self.replay.gather() #Zero this
      packets = super().step(recv)
      self.replay.collect(packets)
      self.nRollouts = len(self.replay)

      done = False
      while not done:
         packets = self.rollout(packets)
         done = self.nRollouts >= self.config.NROLLOUTS

   def send(self):
      grads = self.net.grads()
      blobs = self.blobs
      self.blobs = []
      return grads, blobs

   @runtime
   def step(self, recv):
      self.net.recvUpdate(recv)
      self.collectGrads(recv)
      return self.send()


     
      
      

      
      
