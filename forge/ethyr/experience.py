from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io import stimulus, action, utils

class ExperienceBuffer:
   '''Groups experience into rollouts and
   assembles them into batches'''
   def __init__(self, config):
      self.data = defaultdict(list)
      self.rollouts = {}
      self.nRollouts = 0

   def __len__(self):
      return self.nRollouts

   def collect(self, packets):
      '''Processes a list of serialized experience packets
      
      Args: a list of serialized experience packets
      '''
      for sword, data in enumerate(packets):
         keys, stims, actions, rewards = data
         keys, keyLens = keys

         stims   = stimulus.Dynamic.unbatch(stims)
         actions = action.Dynamic.unbatch(*actions)

         for key, stim, atn, reward in zip(
                  keys, stims, actions, rewards):
            key = key.numpy().astype(np.int).tolist()
            world, tick, annID, entID, serialIdx = key
            key = (world, annID, entID)
            key = tuple(key)
            packet = (stim, atn, reward)

            self.data[key].append(packet)

            if reward == -1:
               self.rollouts[key] = self.data[key]
               del self.data[key]
               self.nRollouts += 1

   def gather(self):
      '''Return rollouts and clear the experience buffer'''
      rollouts = self.rollouts
      self.nRollouts = 0
      self.rollouts = {}
      return rollouts

   def flat(self, rollouts):
      '''Flattens rollouts by removing the time index.
      Useful for batching non recurrent policies

      Args:
         rollouts: A list of rollouts to flatten
      '''
      keys, stims, actions, rewards = [], [], [], []
      for key, rollout in rollouts:
         for val in rollout:
            stim, action, reward = val
            keys.append(key)
            stims.append(stim)
            actions.append(action)
            rewards.append(reward)
      return keys, stims, actions, rewards

   def batch(self, sz):
      '''Wrapper around gather and flat that returns
      flat experience in batches of the specified size

      Args:
         sz: batch size

      Notes:
         The last batch may be smaller than the specified sz
      '''
      data = []
      rollouts = list(self.gather().items())
      while len(rollouts) > 0:
         dat = rollouts[:sz]
         rollouts = rollouts[sz:]
         keys, stims, rawActions, rewards = self.flat(dat)
         stims   = stimulus.Dynamic.batch(stims)
         actions = action.Dynamic.batch(rawActions)
         data.append((keys, stims, rawActions, actions, rewards))
      return data
