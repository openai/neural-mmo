from pdb import set_trace as T
import numpy as np

import time
import ray
import pickle
from collections import defaultdict

from forge.blade.io import stimulus, action, utils

from forge import trinity
from forge.trinity import Base
from forge.trinity.timed import runtime

from forge.ethyr.torch import optim
from forge.ethyr.torch.serial import Serial
from forge.ethyr.rollouts import Rollout

class ExperienceBuffer:
   def __init__(self, config):
      self.data = defaultdict(list)
      self.rollouts = {}
      self.nRollouts = 0
      self.serial = Serial(config)

   def __len__(self):
      return self.nRollouts

   def collect(self, packets):
      for sword, data in enumerate(packets):
         keys, stims, actions, rewards = data

         #Entity ids per example
         keys, keyLens = keys

         #Dict mapping ent ids to tile/entity stims
         stimIDs, stimVals = stims
         stims = stimulus.Dynamic.unbatch(stimIDs, stimVals)
         
         atnTensor, idxTensor, lenTensor = actions
         actions = action.Dynamic.unbatch(atnTensor, idxTensor, lenTensor)
         assert np.all(idxTensor[1] == atnTensor[1]) 
         assert np.all(idxTensor[1] == lenTensor[1]) 

         #lens = lenTensor[1]
         #atnTensor = atnTensor[0].astype(np.int)
         #idxTensor = idxTensor[0].astype(np.int)

         #check = [e[:l] for e, l in zip(atnTensor, lens)]
         #atns = atnTensor[idxTensor]
        
         for key, stim, atn, reward in zip(keys, stims, actions, rewards):
            key = key.numpy().astype(np.int).tolist()
            world, tick, annID, entID = key
            key = (world, annID, entID)
            packet = (stim, atn, reward)
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
      keys, stims, actions, rewards = [], [], [], []
      for key, rollout in rollouts:
         for val in rollout:
            stim, action, reward = val
            keys.append(key)
            stims.append(stim)
            actions.append(action)
            rewards.append(reward)
      return keys, stims, actions, rewards

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
      #self.device = 'cpu:0' 
      self.device = 'cuda:0' 
      self.net = trinity.ANN(config, self.device, mapActions=False).to(self.device)

      self.replay = ExperienceBuffer(config)
      self.blobs  = []

   #Something is fucked here. Step through logic
   def forward(self, keys, stims, actions, rewards):
      rollouts = defaultdict(Rollout)

      rawActions = actions

      stims   = stimulus.Dynamic.batch(stims)
      actions = action.Dynamic.batch(actions)
      atns    = actions[1][0]
      _, outs, vals = self.net(stims, actions, buffered=True)

      atnTensor, idxTensor, lenTensor = actions
      _, lenTensor = lenTensor

      outs = utils.unpack(outs, lenTensor, dim=1)
      #outs, idxs = outs
      #idxs, lens = idxs

      #outs = utils.unpack(outs, lens)
      #atns = utils.unpack(idxs, lens)

      '''
      outs, vals = [], []
      for stim, action in zip(stims, actions):
         _, out, val = self.net(stim, action, buffered=True)
         outs.append(out)
         vals.append(val[0])
      '''

      rets = zip(keys, outs, rawActions, vals, rewards)
      for key, out, atn, val, reward in rets:
         lens, atn = list(zip(*[(len(e), idx) for e, idx in atn]))
         atn = np.array(atn) #* 0 #Actions are not being correctly indexed 
         out = utils.unpack(out, lens)
         rollout = rollouts[key]
         rollout.step(key, out, atn, val, reward)

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


     
      
      

      
      
