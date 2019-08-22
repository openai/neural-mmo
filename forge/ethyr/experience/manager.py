from pdb import set_trace as T
import numpy as np

from itertools import chain 
from collections import defaultdict


from forge.blade.lib.log import BlobLogs

from forge.ethyr.io import Serial
from forge.ethyr.experience import Rollout, Comms, Batcher

class RolloutManager:
   '''Collects and batches rollouts for inference and training

   This is an all-in-one for managing experience. It can be
   used to facillitate fast batched inference, pack/unpack data
   for use on remote machines, and handle network output data
   for use with the ethyr loss/optimizer libraries.
   '''
   def __init__(self):
      self.temp    = defaultdict(Rollout)
      self.outputs = defaultdict(Rollout)
      self.inputs  = {}

      self.logs = BlobLogs()

   @property
   def nUpdates(self):
      return self.logs.nUpdates

   @property
   def nRollouts(self):
      return self.logs.nRollouts

   ### For use on rollout workers ###
   def collectInputs(self, stims):
      '''Collects observation data to internal buffers'''
      #Figure out why stim lens are not matching up. Something
      #to do with newly spawned entities. This was broken in 1.1.
      self.inputs.clear()
      for stim in stims:
         key = stim.key
         rollout = self.temp[key]
         rollout.inputs(stim)

         if stim.done:
            assert key not in self.outputs
            rollout.finish()
            self.outputs[key] = rollout
            del self.temp[key]

            self.logs.blobs.append(rollout.blob)
            self.logs.nRollouts += 1
            self.logs.nUpdates += len(rollout)
 
         else:
            assert key not in self.outputs
            assert key not in self.inputs
            self.inputs[key] = stim


   ### For use on rollout workers ###
   def collectOutputs(self, outputs):
      '''Collects output data to internal buffers'''
      for output in outputs:
         key = output.key

         assert output.key in self.temp
         assert not self.temp[key].done
         self.temp[key].outputs(output)

   def step(self):
      '''Returns log objects of all rollouts.

      Also resets the rollout counter.

      Returns:
         logs: list of blob logging objects
      '''
      logs      = self.logs
      self.logs = BlobLogs()

      outputs      = self.outputs
      self.outputs = defaultdict(Rollout)

      return outputs, logs 

   ### For use on both ###
   def batched(self, nUpdates=None):
      '''Returns flat batches of experience of the specified size

      Notes:
         The last batch may be smaller than the specified sz
      '''
      return Batcher.batched(self.inputs, nUpdates)

