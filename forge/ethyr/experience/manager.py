from pdb import set_trace as T
import numpy as np

from itertools import chain 
from collections import defaultdict


from forge.blade.lib.log import Blob

from forge.ethyr.io import Serial
from forge.ethyr.experience import Rollout, Comms, Batcher

class RolloutManager:
   '''Collects and batches rollouts for inference and training

   This is an all-in-one for managing experience. It can be
   used to facillitate fast batched inference, pack/unpack data
   for use on remote machines, and handle network output data
   for use with the ethyr loss/optimizer libraries.
   '''
   def __init__(self, rollout=Rollout):
      self.partial  = defaultdict(rollout)
      self.complete = defaultdict(rollout)
      self.log = []

      self.nUpdates  = 0
      self.nRollouts = 0

   def reset(self):
      '''Returns log objects of all rollouts.

      Also resets the rollout counter.

      Returns:
         logs: list of blob logging objects
      '''
      assert len(self.complete) == 0
      nUpdates  = self.nUpdates
      nRollouts = self.nRollouts

      self.nUpdates  = 0
      self.nRollouts = 0

      logs = self.log
      self.log = []
      return logs, nUpdates, nRollouts

   ### For use on optimizer ###
   def fill(self, key, out, val, done):
      '''Fill in output/value data needed for the backward pass'''
      key = Serial.nontemporal(key)
      rollout = self.complete[key]
      rollout.fill(key, out, val)

      if done:
         rollout.feather.finish()
         self.log.append(rollout.feather.blob)
         del self.complete[key]

   ### For use on rollout workers ###
   def collectInputs(self, realm, obs, stims):
      '''Collects observation data to internal buffers'''
      for ob, stim, in zip(obs, stims):
         _, key, stim = Serial.inputs(realm, ob, stim)

         self.nUpdates += 1
         iden = Serial.nontemporal(key)

         self.partial[iden].inputs(key, ob, stim)
         self.complete[iden].inputs(key, ob, stim)

   ### For use on rollout workers ###
   def collectOutputs(self, realm, obs, actions, rewards, dones):
      '''Collects output data to internal buffers'''
      self.partial.clear()
      for ob, atn, reward, done in zip(obs, actions, rewards, dones):
         _, key, atn = Serial.outputs(realm, ob, atn)

         iden = Serial.nontemporal(key)
         assert iden in self.complete
         self.complete[iden].outputs(atn, reward, done)

   ### For use on rollout workers ###
   def send(self):
      '''Pack internal buffers for communication across hardware'''
      packet = Comms.send(self.complete)
      self.complete.clear()
      self.reset()
      return packet
   
   ### For use on optimizer ###
   def recv(self, packets):
      '''Unpack communicated data to internal buffers'''
      nUpdates, nRollouts = Comms.recv(
         self.partial, self.complete, packets)
      self.nUpdates  += nUpdates
      self.nRollouts += nRollouts

   ### For use on both ###
   def batched(self, nUpdates, forOptim=False):
      '''Returns flat batches of experience of the specified size

      Notes:
         The last batch may be smaller than the specified sz
      '''
      rollouts = self.complete if forOptim else self.partial
      return Batcher.batched(rollouts, nUpdates, forOptim)

