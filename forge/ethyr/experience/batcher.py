from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.ethyr.io import Stimulus, Action, Serial
from forge.ethyr.io import IO

class Batcher:
   '''Static experience batcher class used internally by RolloutManager'''
   def grouped(rollouts):
      groups = defaultdict(dict)
      for key, rollout in rollouts.items():
         annID, entID = key
         assert key not in groups[annID]
         groups[annID][key] = rollout
      return groups

   def batched(inputs, nUpdates):
      ret, groups = [], Batcher.grouped(inputs)
      for groupKey, group in groups.items():
         group = list(group.items())
         update, updateSz = [], 0 
         for idx, inputs in enumerate(group):
            key, inp = inputs
            if nUpdates is None or updateSz < nUpdates:
               update.append((key, inp))
               updateSz += 1

            #Package a batch of updates
            batchDone = nUpdates is not None and updateSz >= nUpdates
            groupDone = idx == len(group) - 1 
            if batchDone or groupDone:
               packet = Batcher.flat(dict(update))
               ret.append((groupKey, packet))
               update, updateSz = [], 0 

      return ret

   def flat(inputs):
      '''Flattens rollouts by removing the time index.
      Useful for batching non recurrent policies

      Args:
         rollouts: A list of rollouts to flatten
         fullRollouts: whether to batch full rollouts
      '''
      keys  = inputs.keys()
      stims = inputs.values()
      stims, actions = IO.batch(stims)
      return keys, stims, actions

