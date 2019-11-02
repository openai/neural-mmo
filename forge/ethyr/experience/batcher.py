from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.ethyr.io import Stimulus, Action, Serial
from forge.ethyr.io import IO

class Lookup:
   '''Lookup utility for indexing 
   (name, data) pairs'''
   def __init__(self):
      self.data = {}
      self.max = 0

   def add(self, name, idx=None):
      '''Add entries to the table'''
      if idx is None:
         idx = self.max
      self.data[name] =idx 
      self.max += 1

   def __contains__(self, key):
      return key in self.data

   def get(self, idx):
      return self.data[idx]

class Batcher:
   '''Static experience batcher class used internally by RolloutManager'''
   def grouped(rollouts):
      '''Group by population'''
      groups = defaultdict(dict)
      for key, rollout in rollouts.items():
         annID, entID = key
         assert key not in groups[annID]
         groups[annID][key] = rollout
      return groups

   def unique(inputs):
      '''Batch by group key; map to unique set of obs '''
      data = defaultdict(lambda: defaultdict(list))
      objCounts = defaultdict(int)
      objLookup = Lookup()
      for key, inp in inputs.items():
         stim, atn = inp.stim, inp.action
   
         for group, objs in stim.items():
            names, vals = objs
            for idx, objID in enumerate(names):
               if objID in objLookup:
                  continue

               objLookup.add(objID)#, idx=objCounts[group])
               #objCounts[group] += 1
               for attr, val in vals.items():
                  #Check idx
                  data[group][attr].append(val[0][idx])

            inp.stim[group] = inp.stim[group][0]
            
      return data, objLookup

   def batched(inputs, nUpdates):
      '''Batch by group key to maximum fixed size'''
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
               update = dict(update)
               keys   = update.keys()
               stims  = update.values()
               stims, actions = IO.batch(stims)
               packet = (keys, stims, actions)
               ret.append((groupKey, packet))
               update, updateSz = [], 0 

      return ret
