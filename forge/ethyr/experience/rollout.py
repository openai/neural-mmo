from pdb import set_trace as T

from collections import defaultdict
import numpy as np

from forge.blade.lib.log import Blob

class Output:
   def __init__(self, atnArgKey, atnLogits, atnIdx, value):
      '''Data structure specifying a chosen action

      Args:
         atnArgKey : Action-Argument formatted string                           
         atnLogits : Action logits                                              
         atnsIdx   : Argument indices sampled from logits                       
         value     : Value function prediction  
      '''
      self.atnArgKey = atnArgKey
      self.atnLogits = atnLogits
      self.atnIdx    = atnIdx      
      self.value     = value

class Rollout:
   def __init__(self, config):
      '''Rollout object used internally by RolloutManager

      Args:
         config: A configuration object
      '''
      self.actions = defaultdict(list)
      self.values  = []
      self.rewards = []

      self.done = False
      self.time = -1

      #Logger
      self.config  = config
      self.blob    = None

   def __len__(self):
      '''Length of a rollout

      Returns:
         lifetime: Number of timesteps the agent has survived
      '''
      return self.blob.lifetime

   def inputs(self, reward, key):
      '''Collects input data to internal buffers

      Args:
         reward : The reward received by the agent for its last action
         key    : The ID associated with the agent
      '''
      if reward is not None:
         self.rewards.append(reward)

      if self.blob is None:
         annID, entID = key
         self.blob = Blob(entID, annID)

      self.time += 1
      self.blob.update()

   def outputs(self, atnArgKey, atnLogits, atnIdx, value):
      '''Collects output data to internal buffers

      Args:
         atnArgKey : Action-Argument formatted string                           
         atnLogits : Action logits                                              
         atnsIdx   : Argument indices sampled from logits                       
         value     : Value function prediction  
      '''
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
         gamma: Reward discount factor

      Returns:
         rewards: Discounted list of rewards
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
