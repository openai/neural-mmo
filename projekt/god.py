from pdb import set_trace as T
import numpy as np

import time
import ray
import pickle
from collections import defaultdict

import projekt


from forge import trinity
from forge.trinity.timed import runtime

from forge.ethyr.io import Stimulus, Action, utils
from forge.ethyr.torch import optim
from forge.ethyr.experience import RolloutManager

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

      self.manager = RolloutManager()

      self.net = projekt.ANN(
               config).to(self.config.DEVICE)

   @runtime
   def step(self, recv):
      '''Broadcasts updated weights to the core level
      Sword rollout workers. Runs rollout workers'''
      self.net.recvUpdate(recv)
      self.rollouts(recv)

      #Send update
      grads = self.net.grads()
      logs, nUpdates, nRollouts  = self.manager.reset()
      return grads, logs, nUpdates, nRollouts

   def rollouts(self, recv):
      '''Runs rollout workers while asynchronously
      computing gradients over available experience'''
      self.nRollouts, done = 0, False
      while not done:
         packets = super().distrib(recv) #async rollout workers
         self.processRollouts()          #intermediate gradient computatation
         packets = super().sync(packets) #sync next batches of experience
         self.manager.recv(packets)

         done = self.manager.nUpdates >= self.config.OPTIMUPDATES
      self.processRollouts() #Last batch of gradients

   def processRollouts(self):
      '''Runs minibatch forwards/backwards
      over all available experience'''
      for batch in self.manager.batched(
            self.config.OPTIMBATCH, forOptim=True):
         rollouts = self.forward(*batch)
         self.backward(rollouts)

   def forward(self, pop, rollouts, data):
      '''Recompute forward pass and assemble rollout objects'''
      keys, _, stims, rawActions, actions, rewards, dones = data
      _, outs, vals = self.net(pop, stims, atnArgs=actions)

      #Unpack outputs
      atnTensor, idxTensor, atnKeyTensor, lenTensor = actions
      lens, lenTensor = lenTensor
      atnOuts = utils.unpack(outs, lenTensor, dim=1)

      #Collect rollouts
      for key, out, atn, val, reward, done in zip(
            keys, outs, rawActions, vals, rewards, dones):

         atnKey, lens, atn = list(zip(*[(k, len(e), idx) 
            for k, e, idx in atn]))

         atn = np.array(atn)
         out = utils.unpack(out, lens)

         self.manager.fill(key, (atnKey, atn, out), val, done)

      return rollouts

   def backward(self, rollouts):
      '''Compute backward pass and logs from rollout objects'''
      reward, val, pg, valLoss, entropy = optim.backward(
            rollouts, valWeight=0.25,
            entWeight=self.config.ENTROPY, device=self.config.DEVICE)

    
