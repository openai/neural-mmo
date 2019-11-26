from pdb import set_trace as T
import numpy as np

from itertools import chain 
from collections import defaultdict


from forge.blade.lib.log import BlobSummary

from forge.ethyr.io import Serial
from forge.ethyr.experience import Rollout, Batcher

class RolloutManager:
   '''Collects and batches rollouts for inference and training'''
   def __init__(self):
      self.temp    = defaultdict(Rollout)
      self.outputs = defaultdict(Rollout)
      self.logs    = BlobSummary()

   @property
   def nUpdates(self):
      return self.logs.nUpdates

   @property
   def nRollouts(self):
      return self.logs.nRollouts

   def collectInputs(self, stims):
      '''Collects observation data to internal buffers'''
      #Finish rollout
      for key in stims.dones:
         assert key not in self.outputs

         rollout           = self.temp[key]
         rollout.finish()

         self.outputs[key] = rollout
         del self.temp[key]

         self.logs.blobs.append(rollout.blob)
         self.logs.nRollouts += 1
         self.logs.nUpdates += len(rollout)

      #Update inputs 
      for key, reward in zip(stims.keys, stims.rewards ):
         assert key not in self.outputs
         rollout = self.temp[key]
         rollout.inputs(reward, key)

   def collectOutputs(self, atnArg, keys, atns, atnsIdx, values):
      '''Collects output data to internal buffers'''
      for key, atn, atnIdx, val in zip(keys, atns, atnsIdx, values):
         assert key in self.temp
         assert not self.temp[key].done
         self.temp[key].outputs(atnArg, atn, atnIdx, val)

   def step(self):
      '''Returns log objects of all rollouts.

      Also resets the rollout counter.

      Returns:
         outputs, logs: rolloutdict, list of blob logging objects
      '''
      logs      = self.logs
      self.logs = BlobSummary()

      outputs      = self.outputs
      self.outputs = defaultdict(Rollout)

      return outputs, logs 
