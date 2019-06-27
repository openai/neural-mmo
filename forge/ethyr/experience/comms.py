from pdb import set_trace as T
import numpy as np

from itertools import chain 
from collections import defaultdict

from forge.ethyr.io import Stimulus, Action, Serial

class Comms:
   '''Static communication class used internally by RolloutManager'''
   def send(rollouts):
      '''Pack rollouts from workers for optim server'''
      keys, stims, actions, rewards, dones = [], [], [], [], []
      for key, rollout in rollouts.items():
         keys    += rollout.keys
         stims   += rollout.stims
         actions += rollout.actions
         rewards += rollout.rewards
         dones   += rollout.dones

      stims   = Stimulus.batch(stims)
      actions = Action.batch(actions)
      keys = np.stack(keys)

      return keys, stims, actions, rewards, dones

   def recv(partial, full, packets):
      '''Unpack rollouts from workers on optim server
      
      Args: 
         partial: A defaultdict of partially complete rollouts
         full: A defaultdict of complete rollouts
         packets: a list of serialized experience packets
      '''
      nNew = 0
      for sword, data in enumerate(packets):
         keys, stims, actions, rewards, dones = data

         stims   = Stimulus.unbatch(stims)
         actions = Action.unbatch(*actions)

         for iden, stim, atn, reward, done in zip(
               keys, stims, actions, rewards, dones):

            key = Serial.nontemporal(iden)

            partial[key].inputs(iden, None, stim)
            partial[key].outputs(atn, reward, done)

            if partial[key].done:
               assert key not in full
               full[key] = partial[key]

               del partial[key]
               nNew += 1
      return nNew

