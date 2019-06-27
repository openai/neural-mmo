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
      self.feather = Feather()

   def __len__(self):
      assert self.time == len(self.stims)
      return self.time

   def discount(self, gamma=0.99):
      '''Applies standard gamma discounting to the given trajectory
      
      Args:
         rewards: List of rewards
         gamma: Discount factor

      Returns:
         Discounted list of rewards

      Notes:
         You can override Rollout to modify the discount algorithm
      '''
      rets, N = [], len(self.rewards)
      discounts = np.array([gamma**i for i in range(N)])
      rewards = np.array(self.rewards)
      for idx in range(N):
         rets.append(sum(rewards[idx:]*discounts[:N-idx]))
      return rets

   def fill(self, key, out, val):
      '''Add in data needed for backwards pass'''
      self.outs.append(out) 
      self.vals.append(val) 

      self.feather.scrawl(key)
      self.feather.value(val)

   def inputs(self, iden, ob, stim):
      '''Process observation data'''
      self.obs.append(ob)
      self.stims.append(stim)
      self.keys.append(iden)
      self.time += 1

   def outputs(self, atn, reward, done):
      '''Process output action/reward/done data'''
      self.actions.append(atn)
      self.rewards.append(reward)
      self.dones.append(done)

      self.done = done

      if done:
         self.finish()

      self.feather.reward(reward)

   def finish(self):
      '''Called internally once the full rollout has been collected'''
      assert self.rewards[-1] == -1
      self.returns = self.discount()
      self.lifespan = len(self.rewards)

class Feather:
   '''Internal logger used by Rollout. Due for a rewrite.'''
   def __init__(self):
      self.expMap = set()
      self.blob = Blob()

   def scrawl(self, iden):
      '''Write logs from one time step

      Args:
         iden: The unique ID used in serialization
      '''
      world, annID, entID, _ = iden
      self.blob.entID = entID
      self.blob.annID = annID
      self.blob.world = world
      
      #tile = self.tile(stim)
      #self.move(tile, ent.pos)
      #self.action(arguments, atnArgs)

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

   def reward(self, reward):
      self.blob.reward.append(reward)

   def value(self, value):
      self.blob.value.append(float(value))

   def finish(self):
      self.blob.finish()

