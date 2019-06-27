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
      self.updates = defaultdict(rollout)
      self.rollouts = defaultdict(rollout)
      self.done = {}
      self.log = []

      self.nUpdates  = 0
      self.nRollouts = 0

   def logs(self):
      '''Returns log objects of all rollouts.

      Also resets the rollout counter.

      Returns:
         logs: list of blob logging objects
      '''
      assert len(self.done) == 0
      self.nRollouts = 0
      logs = self.log
      self.log = []
      return logs

   def collectInputs(self, realm, obs, stims):
      '''Collects observation data to internal buffers'''

      for ob, stim, in zip(obs, stims):
         _, key, stim = Serial.inputs(realm, ob, stim)

         self.nUpdates += 1
         iden = Serial.nontemporal(key)

         self.rollouts[iden].inputs(key, ob, stim)

         assert iden not in self.updates
         self.updates[iden].inputs(key, ob, stim)

   def collectOutputs(self, realm, obs, actions, rewards, dones):
      '''Collects output data to internal buffers'''
      self.updates.clear()
      for ob, atn, reward, done in zip(obs, actions, rewards, dones):
         _, key, atn = Serial.outputs(realm, ob, atn)

         iden = Serial.nontemporal(key)
         self.rollouts[iden].outputs(atn, reward, done)

   def fill(self, key, out, val, done):
      '''Fill in output/value data needed for the backward pass'''
      key = Serial.nontemporal(key)
      rollout = self.done[key]
      rollout.fill(key, out, val)

      if done:
         rollout.feather.finish()
         self.log.append(rollout.feather.blob)
         del self.done[key]

   def send(self):
      '''Pack internal buffers for communication across hardware'''
      return Comms.send(self.rollouts)
   
   def recv(self, packets):
      '''Unpack communicated data to internal buffers'''
      self.nRollouts += Comms.recv(
         self.rollouts, self.done, packets)

   def batched(self, batchSize, fullRollouts=True):
      '''Returns flat batches of experience of the specified size

      Notes:
         The last batch may be smaller than the specified sz
      '''
      if fullRollouts:
         rollouts = self.done
      else:
         rollouts = self.updates
      return Batcher.batched(rollouts, batchSize, fullRollouts)

