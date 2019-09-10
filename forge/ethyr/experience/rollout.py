from pdb import set_trace as T
import numpy as np

from itertools import chain 
from collections import defaultdict

from forge.blade.lib.log import Blob

class Rollout:
   '''Rollout object used internally by RolloutManager'''
   def __init__(self):
      self.keys, self.obs      = [], []
      self.stims, self.actions = [], []
      self.rewards, self.dones = [], []
      self.outs, self.vals     = [], []

      self.done = False
      self.time = 0

      #Logger
      self.blob = None

   def __len__(self):
      #assert self.time == len(self.stims)
      return self.blob.lifetime

   def discount(self, gamma=0.95):
      '''Applies standard gamma discounting to the given trajectory
      
      Args:
         rewards: List of rewards
         gamma: Discount factor

      Returns:
         Discounted list of rewards
      '''
      rets, N = [], len(self.rewards)
      discounts = np.array([gamma**i for i in range(N)])
      rewards = np.array(self.rewards)
      for idx in range(N):
         rets.append(sum(rewards[idx:]*discounts[:N-idx]))
      return rets

   def inputs(self, inputs):
      '''Process observation data'''
      self.stims.append(inputs)
      
      reward, done = inputs.reward, inputs.done
      
      if reward is not None:
         self.rewards.append(reward)
      if done is not None:
         self.dones.append(done)
      if done:
         self.done = True

      if self.blob is None:
         self.blob = Blob(inputs.entID, inputs.annID)

   def outputs(self, output):
      '''Process output action/reward/done data'''
      self.actions.append(output.action)
      self.vals.append(output.value)
      self.outs.append(output.out)

      self.time += 1
      self.blob.update()

   def finish(self):
      '''Called internally once the full rollout has been collected'''
      assert self.rewards[-1] == -1
      self.returns  = self.discount()
      self.lifespan = len(self.rewards)

      self.blob.value  = np.mean([float(e) for e in self.vals])
