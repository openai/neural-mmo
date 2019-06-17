from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io import stimulus, action, utils

class ExperienceBuffer:
   def __init__(self, config):
      self.data = defaultdict(list)
      self.rollouts = {}
      self.nRollouts = 0

   def __len__(self):
      return self.nRollouts

   def collect(self, packets):
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
      rollouts = self.rollouts
      self.nRollouts = 0
      self.rollouts = {}
      return rollouts

   def flat(self, rollouts):
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
      data = []
      rollouts = list(self.gather().items())
      while len(rollouts) > 0:
         dat = rollouts[:sz]
         rollouts = rollouts[sz:]
         dat = self.flat(dat)
         data.append(dat)
      return data
