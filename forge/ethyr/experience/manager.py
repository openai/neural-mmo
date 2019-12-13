from pdb import set_trace as T
import numpy as np

from itertools import chain 
from collections import defaultdict

from forge.blade.io.serial import Serial
from forge.blade.lib.log import BlobSummary

from forge.ethyr.experience import Rollout

class RolloutManager:
   def __init__(self, config):
      '''Experience batcher for inference and training
                                                                              
      Args:                                                                   
         config: A Configuration object                                       
      '''  
      self.inputs  = defaultdict(lambda: Rollout(config))
      self.outputs = defaultdict(lambda: Rollout(config))
      self.logs    = BlobSummary()

   @property
   def nUpdates(self):
      '''Number of experience steps collected thus far

      Returns:
         n: The number of updates
      '''
      return self.logs.nUpdates

   @property
   def nRollouts(self):
      '''Number of full trajectories collected thus far

      Returns:
         n: The number of rollouts
      '''
      return self.logs.nRollouts

   def collectInputs(self, stims):
      '''Collects observation data to internal buffers

      Args:
         stims: Input data to batch
      '''
      #Finish rollout
      for key in stims.dones:
         assert key not in self.outputs

         #Already cleared as a partial traj
         if key not in self.inputs:
            continue

         rollout           = self.inputs[key]
         self.outputs[key] = rollout

         rollout.finish()
         del self.inputs[key]

         self.logs.add([rollout.blob])

      #Update inputs 
      for key, reward in zip(stims.keys, stims.rewards):
         assert key not in self.outputs
         rollout = self.inputs[key]
         rollout.inputs(reward, key)

   def collectOutputs(self, atnArg, keys, atns, atnsIdx, values):
      '''Collects output data to internal buffers

      Args:
         atnArg  : Action-Argument formatted string
         keys    : Identifiers for each agent
         atns    : Action logits
         atnsIdx : Argument indices sampled from logits
         values  : Value function prediction
      '''
      for key, atn, atnIdx, val in zip(keys, atns, atnsIdx, values):
         assert key in self.inputs
         assert not self.inputs[key].done
         self.inputs[key].outputs(atnArg, atn, atnIdx, val)

   def step(self):
      '''Aggregates rollouts and logs, resetting internal buffers

      Returns:
         outputs : Dictionary of rollouts
         logs    : List of logging Blob objects
      '''
      logs      = self.logs
      self.logs = BlobSummary()

      outputs      = self.outputs
      self.outputs = defaultdict(Rollout)

      return outputs, logs 
