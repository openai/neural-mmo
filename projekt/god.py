from pdb import set_trace as T
import numpy as np

import time
import ray
import pickle
from collections import defaultdict

import projekt
from forge.blade.io import stimulus, action, utils

from forge import trinity
from forge.trinity.timed import runtime

from forge.ethyr.torch import optim
from forge.ethyr.torch.experience import ExperienceBuffer
from forge.ethyr.rollouts import RolloutManager

import torch

@ray.remote(num_gpus=1)
class God(trinity.God):
   '''Server level God API demo

   This server level optimizer node aggregates experience
   across all core level rollout worker nodes. It
   uses the aggregated experience compute gradients.

   This is effectively a lightweight variant of the
   Rapid computation model, with the potential notable
   difference that we also recompute the forward pass
   from small observation buffers rather than 
   communicating large activation tensors.

   This demo builds up the ExperienceBuffer utility, 
   which handles rollout batching.'''

   def __init__(self, trin, config, args, idx):
      '''Initializes a model and relevent utilities'''
      super().__init__(trin, config, args, idx)
      self.config, self.args = config, args

      self.blobs = []
      self.replay = ExperienceBuffer(config)

      self.net = projekt.ANN(
               config).to(self.config.DEVICE)

   @runtime
   def step(self, recv):
      '''Broadcasts updated weights to the core level
      Sword rollout workers. Runs rollout workers'''
      self.net.recvUpdate(recv)
      self.rollouts(recv)

      #Send update
      grads      = self.net.grads()
      blobs      = self.blobs
      self.blobs = []
      return grads, blobs

   def rollouts(self, recv):
      '''Runs rollout workers while asynchronously
      computing gradients over available experience'''
      self.nRollouts, done = 0, False
      while not done:
         packets = super().distrib(recv) #async rollout workers
         self.processRollouts()          #intermediate gradient computatation
         packets = super().sync(packets) #sync next batches of experience
         self.replay.collect(packets)

         self.nRollouts += len(self.replay)
         done = self.nRollouts >= self.config.NROLLOUTS
      self.processRollouts() #Last batch of gradients

   def processRollouts(self):
      '''Runs minibatch forwards/backwards
      over all available expereince'''
      batches = self.replay.batch(self.config.BATCH)
      for batch in batches:
         rollouts = self.forward(*batch)
         self.backward(rollouts)

   #We have two issue here.
   #First, computation time is increasing on Server with more client nodes,
   #despite ingesting the same amount of data. This may be a logging bug
   #Second, even very small networks have a long execution time due to 
   #a large number of operations. We aren't leveraging the GPU and getting
   #good speedups. Ideal is that with nrealm=1, we get x time on GPU and 
   #10X time on CPU. Then we use 10 cpus to saturate the GPU with the same
   #amount of data per block in 1/10 time
   def forward(self, keys, stims, rawActions, actions, rewards):
      '''Recompute forward pass and assemble rollout objects'''
      _, outs, vals = self.net(stims, atnArgs=actions)

      #Unpack outputs
      atnTensor, idxTensor, atnKeyTensor, lenTensor = actions
      lens, lenTensor = lenTensor
      atnOuts = utils.unpack(outs, lenTensor, dim=1)

      #Collect rollouts
      rollouts = RolloutManager()
      for key, out, atn, val, reward in zip(
            keys, outs, rawActions, vals, rewards):
         atnKey, lens, atn = list(zip(*[(k, len(e), idx) for k, e, idx in atn]))
         atn = np.array(atn)
         out = utils.unpack(out, lens)

         rollouts[key].step(key, atnKey, out, atn, val, reward)

      rollouts.finish()
      return rollouts

   def backward(self, rollouts):
      '''Compute backward pass and logs from rollout objects'''
      reward, val, pg, valLoss, entropy = optim.backward(
            rollouts, valWeight=0.25,
            entWeight=self.config.ENTROPY, device=self.config.DEVICE)
      self.blobs += rollouts.logs()

    
