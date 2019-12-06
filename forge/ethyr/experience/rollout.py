from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.lib.log import Blob

class Output:
   def __init__(self, atnArgKey, atnLogits, atnIdx, value):
      self.atnArgKey = atnArgKey
      self.atnLogits = atnLogits
      self.atnIdx    = atnIdx      
      self.value     = value

class Rollout:
   '''Rollout object used internally by RolloutManager'''
   def __init__(self, config):
      self.actions = defaultdict(list)
      self.values  = []
      self.rewards = []

      self.done = False
      self.time = -1

      #Logger
      self.config  = config
      self.blob    = None

   def __len__(self):
      return self.blob.lifetime

   def inputs(self, reward, key):
      '''Process observation data'''
      if reward is not None:
         self.rewards.append(reward)

      if self.blob is None:
         annID, entID = key
         self.blob = Blob(entID, annID)

      self.time += 1
      self.blob.update()

   def outputs(self, atnArgKey, atnLogits, atnIdx, value):
      '''Process output actions and values'''
      output = Output(atnArgKey, atnLogits, atnIdx, value)
      self.actions[self.time].append(output)
      self.values.append(value)

   def finish(self):
      '''Called internally once the full rollout has been collected'''
      self.rewards.append(-1)

      self.returns     = self.discount(self.config.DISCOUNT)
      self.lifespan    = len(self.rewards)
      self.blob.value  = np.mean([float(e) for e in self.values])

   def discount(self, gamma):
      '''Applies standard gamma discounting to the given trajectory
      
      Args:
         rewards: List of rewards
         gamma: Discount factor

      Returns:
         Discounted list of rewards
      '''
      rets, N   = [], len(self.rewards)
      discounts = np.array([gamma**i for i in range(N)])
      rewards   = np.array(self.rewards)

      for idx in range(N):
         R_i = sum(rewards[idx:]*discounts[:N-idx])
         for out in self.actions[idx]:
            out.returns = R_i 
         
         rets.append(R_i)

      return rets


