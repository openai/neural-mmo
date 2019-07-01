from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.ethyr.io import Stimulus, Action, Serial

class Batcher:
   '''Static experience batcher class used internally by RolloutManager'''
   def grouped(rollouts):
      groups = defaultdict(dict)
      for key, rollout in rollouts.items():
         groups[Serial.population(key)][key] = rollout
      return groups.items()

   
   def batched(rollouts, nUpdates, fullRollouts):
      ret, groups = [], Batcher.grouped(rollouts)
      for groupKey, group in Batcher.grouped(rollouts):
         group = list(group.items())
         update, updateSz = [], 0 
         for idx, rollout in enumerate(group):
            key, rollout = rollout
            if updateSz < nUpdates:
               updateSz += len(rollout)
               update.append((key, rollout))
            if updateSz >= nUpdates or idx == len(group) - 1:
               rolls = dict(update)
               packet = Batcher.flat(rolls, fullRollouts)
               ret.append((groupKey, rolls, packet))
               update, updateSz = [], 0 

         '''
         for idx in range(0, len(group), nUpdates):
            rolls = dict(group[idx:idx+nUpdates])
            packet = Batcher.flat(rolls, fullRollouts)
            ret.append((groupKey, rolls, packet))
         '''

      return ret

   def flat(rollouts, fullRollouts):
      '''Flattens rollouts by removing the time index.
      Useful for batching non recurrent policies

      Args:
         rollouts: A list of rollouts to flatten
         fullRollouts: whether to batch full rollouts
      '''
      tick = 0
      keys, obs = [], []
      rawStims, rawActions = [], []
      stims, actions, rewards, dones = [], [], [], []

      for key, rollout in rollouts.items():
         keys     += rollout.keys
         stims    += rollout.stims

         if fullRollouts:
            rawActions += rollout.actions
            rewards    += rollout.returns
            dones      += rollout.dones
         else:
            rawStims   += rollout.obs

      stims   = Stimulus.batch(stims)

      if fullRollouts:
         actions = Action.batch(rawActions)

      return keys, rawStims, stims, rawActions, actions, rewards, dones

