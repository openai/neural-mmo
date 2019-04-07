from pdb import set_trace as T
from itertools import chain 
from collections import defaultdict
import numpy as np

from forge.blade.lib.log import Blob

#Untested function
def discountRewards(rewards, gamma=0.99):
   rets, N = [], len(rewards)
   discounts = np.array([gamma**i for i in range(N)])
   rewards = np.array(rewards)
   for idx in range(N): rets.append(sum(rewards[idx:]*discounts[:N-idx]))
   return rets

def sumReturn(rewards):
   return [sum(rewards) for e in rewards]

def mergeRollouts(rollouts):
   outs = {'value': [], 'return': [], 
         'action': defaultdict(lambda: defaultdict(list))}
   for rollout in rollouts:
      for idx in range(rollout.time):
         aarg = rollout.atnArgs[idx]
         val  = rollout.vals[idx]
         ret  = rollout.returns[idx]

         outs['value'].append(val)
         outs['return'].append(ret)

         for k, out in aarg.items():
            atn, idx = out
            outk = outs['action'][k]
            outk['atns'].append(atn) 
            outk['idxs'].append(idx)
            outk['vals'].append(val)
            outk['rets'].append(ret)
   return outs

class Rollout:
   def __init__(self, returnf=discountRewards):
      self.atnArgs = []
      self.vals = []
      self.rewards = []
      self.pop_rewards = []
      self.returnf = returnf
      self.feather = Feather()
      self.time = 0

   def step(self, atnArgs, val, reward):
      self.atnArgs.append(atnArgs)
      self.vals.append(val)
      self.rewards.append(reward)
      self.time += 1

   def finish(self):
      self.rewards[-1] = -1
      self.returns = self.returnf(self.rewards)
      self.lifespan = len(self.rewards)
      self.feather.finish()

#Rollout logger
class Feather:
   def __init__(self):
      self.expMap = set()
      self.blob = Blob()

   def scrawl(self, stim, ent, val, reward):
      self.blob.annID = ent.annID
      tile = self.tile(stim)
      self.move(tile, ent.pos)
      #self.action(arguments, atnArgs)
      self.stats(val, reward)

   def tile(self, stim):
      R, C = stim.shape
      rCent, cCent = R//2, C//2
      tile = stim[rCent, cCent]
      return tile

   def action(self, arguments, atnArgs):
      move, attk = arguments
      moveArgs, attkArgs, _ = atnArgs
      moveLogits, moveIdx = moveArgs
      attkLogits, attkIdx = attkArgs

   def move(self, tile, pos):
      tile = type(tile.state)
      if pos not in self.expMap:
         self.expMap.add(pos)
         self.blob.unique[tile] += 1
      self.blob.counts[tile] += 1

   def stats(self, value, reward):
      self.blob.reward.append(reward)
      self.blob.value.append(float(value))

   def finish(self):
      self.blob.finish()

