from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.lib.log import Blob

class Output:
   def __init__(self, atnArgKey, atnLogits, atnIdx):
      self.atnArgKey = atnArgKey
      self.atnLogits = atnLogits
      self.atnIdx    = atnIdx      

class Rollout:
   '''Rollout object used internally by RolloutManager'''
   def __init__(self):
      self.actions = defaultdict(list)
      self.values  = []
      self.rewards = []

      self.done = False
      self.time = -1

      #Logger
      self.blob = None

   def __len__(self):
      return self.blob.lifetime

   def inputs(self, reward, key):
      '''Process observation data'''
      self.time += 1

      if reward is not None:
         self.rewards.append(reward)

      if self.blob is None:
         annID, entID = key
         self.blob = Blob(entID, annID)

      self.blob.update()

   def outputs(self, atnArgKey, atnLogits, atnIdx, value):
      '''Process output actions and values'''
      actions = self.actions[self.time]

      if len(actions) == 0:
         self.values.append(value)

      output = Output(atnArgKey, atnLogits, atnIdx)
      actions.append(output)

   def finish(self):
      '''Called internally once the full rollout has been collected'''
      self.rewards.append(-1)

      self.returns     = self.discount()
      self.lifespan    = len(self.rewards)
      self.blob.value  = np.mean([float(e) for e in self.values])

   def discount(self, gamma=0.95):
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
         rets.append(sum(rewards[idx:]*discounts[:N-idx]))

      return rets


