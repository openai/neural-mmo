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
         key = rollout.keys[idx]
         out = rollout.outs[idx]
         atn = rollout.atns[idx]
         val = rollout.vals[idx]
         ret = rollout.returns[idx]

         outs['value'].append(val)
         outs['return'].append(ret)

         for k, o, a in zip(key, out, atn):
            k = tuple(k)
            outk = outs['action'][k]
            outk['atns'].append(o) 
            outk['idxs'].append(a)
            outk['vals'].append(val)
            outk['rets'].append(ret)
   return outs

class Rollout:
   '''Rollout class'''
   def __init__(self, returnf=discountRewards):
      '''Rollout object
        
      Args:
         return function hook

      '''         

      self.keys = []
      self.outs = []
      self.atns = []
      self.vals = []
      self.rewards = []
      self.pop_rewards = []
      self.returnf = returnf
      self.feather = Feather()
      self.time = 0

   def step(self, iden, key, out, atn, val, reward):
      self.keys.append(key)
      self.outs.append(out)
      self.atns.append(atn)
      self.vals.append(val)
      self.rewards.append(reward)
      self.time += 1

      self.feather.scrawl(iden, atn, val, reward)

   def finish(self):
      assert self.rewards[-1] == -1
      self.returns = self.returnf(self.rewards)
      self.lifespan = len(self.rewards)
      self.feather.finish()

#Rollout logger
class Feather:
   def __init__(self):
      self.expMap = set()
      self.blob = Blob()

   def scrawl(self, iden, atn, val, reward):
      world, annID, entID = iden
      self.blob.entID = entID
      self.blob.annID = annID
      self.blob.world = world
      
      #tile = self.tile(stim)
      #self.move(tile, ent.pos)
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
         if tile in self.blob.unique:
            self.blob.unique[tile] += 1
      if tile in self.blob.counts:
         self.blob.counts[tile] += 1

   def stats(self, value, reward):
      self.blob.reward.append(reward)
      self.blob.value.append(float(value))

   def finish(self):
      self.blob.finish()

