from pdb import set_trace as T
import numpy as np

import time
import ray
import pickle
from collections import defaultdict

from forge.blade.io import stimulus, action, utils

from forge import trinity
from forge.trinity import Base
from forge.trinity.experience import ExperienceBuffer
from forge.trinity.timed import runtime

from forge.ethyr.torch import optim
from forge.ethyr.rollouts import Rollout

@ray.remote(num_gpus=1)
class God(Base.God):
   def __init__(self, trin, config, args):
      super().__init__(trin, config, args)
      self.config, self.args = config, args

      self.net = trinity.ANN(
            config).to(self.config.DEVICE)

      self.replay = ExperienceBuffer(config)
      self.blobs  = []

   @runtime
   def step(self, recv):
      self.net.recvUpdate(recv)
      self.collectGrads(recv)
      return self.send()

   def collectGrads(self, recv):
      self.replay.gather() #Zero this
      packets = super().step(recv)
      self.replay.collect(packets)
      self.nRollouts = len(self.replay)

      done = False
      while not done:
         packets = self.rollout(packets)
         done = self.nRollouts >= self.config.NROLLOUTS

   def rollout(self, packets, recv=None):
      packets = super().distrib(recv)
      self.processReplay()
      packets = super().sync(packets)

      self.replay.collect(packets)
      self.nRollouts += len(self.replay)
      return packets

   def processReplay(self):
      batches = self.replay.batch(self.config.BATCH)
      for batch in batches:
         rollouts = self.forward(*batch)
         self.backward(rollouts)

   def send(self):
      grads = self.net.grads()
      blobs = self.blobs
      self.blobs = []
      return grads, blobs

   def forward(self, keys, stims, actions, rewards):
      rollouts, rawActions = defaultdict(Rollout), actions

      stims   = stimulus.Dynamic.batch(stims)
      T()
      actions = action.Dynamic.batch(actions)
      _, outs, vals = self.net(stims, atnArgs=actions)

      #Unpack outputs
      atnTensor, idxTensor, atnKeyTensor, lenTensor = actions
      lens, lenTensor = lenTensor
      atnOuts = utils.unpack(outs, lenTensor, dim=1)

      #Collect rollouts
      rets = zip(keys, outs, rawActions, vals, rewards)
      for key, out, atn, val, reward in rets:
         atnKey, lens, atn = list(zip(*[(k, len(e), idx) for k, e, idx in atn]))
         atn = np.array(atn)
         out = utils.unpack(out, lens)

         rollout = rollouts[key]
         rollout.step(key, atnKey, out, atn, val, reward)

      for key, rollout in rollouts.items():
         rollout.finish()

      return rollouts

   def backward(self, rollouts):
      reward, val, pg, valLoss, entropy = optim.backward(
            rollouts, valWeight=0.25,
            entWeight=self.config.ENTROPY, device=self.config.DEVICE)

      '''
      print('Reward : ', float(reward))
      print('Value  : ', float(val))
      print('PolLoss: ', float(pg))
      print('ValLoss: ', float(valLoss))
      print('Entropy: ', float(entropy))
      '''
      self.blobs += [r.feather.blob for r in rollouts.values()]

    
