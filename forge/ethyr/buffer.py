from pdb import set_trace as T
import numpy as np

from itertools import chain 
from collections import defaultdict

from forge.blade.io import stimulus, action, utils
from forge.blade.lib.log import Blob
from forge.ethyr.torch import utils

def gammaDiscount(rewards, gamma=0.99):
   '''Applies standard discounting to the given trajectory
   
   Args:
      rewards: List of rewards
      gamma: Discount factor

   Returns:
      Discounted list of rewards
   '''
   rets, N = [], len(rewards)
   discounts = np.array([gamma**i for i in range(N)])
   rewards = np.array(rewards)
   for idx in range(N):
      rets.append(sum(rewards[idx:]*discounts[:N-idx]))
   return rets

class Batcher:
   def grouped(rollouts):
      groups = defaultdict(dict)
      for key, rollout in rollouts.items():
         groups[Serial.population(key)][key] = rollout
      return groups.items()
   
   def batched(rollouts, batchSize):
      ret, groups = [], Batcher.grouped(rollouts)
      for _, group in Batcher.grouped(rollouts):
         group = list(group.items())
         for idx in range(0, len(group), batchSize):
            rolls = dict(group[idx:idx+batchSize])
            packet = Batcher.flat(rolls)

            lens, keys, rawStims, rawActions, rewards, dones = packet
            stims   = stimulus.Dynamic.batch(rawStims)
            actions = action.Dynamic.batch(rawActions)
            data = (keys, stims, rawActions, actions, rewards, dones)

            ret.append((rolls, data))
      return ret

   def flat(rollouts):
      '''Flattens rollouts by removing the time index.
      Useful for batching non recurrent policies

      Args:
         rollouts: A list of rollouts to flatten
      '''

      tick, lens = 0, []
      keys, stims, actions, rewards, dones = [], [], [], [], []
      for key, rollout in rollouts.items():
         keys    += rollout.keys
         stims   += rollout.stims
         actions += rollout.actions
         rewards += rollout.returns
         dones   += rollout.dones
         lens.append(len(rollout))
      return lens, keys, stims, actions, rewards, dones

class Serial:
   KEYLEN = 5
   def key(key, iden):
      from forge.blade.entity import Player
      from forge.blade.core.tile import Tile

      ret = key.serial
      if isinstance(key, type):
         if issubclass(key, action.Node):
            ret += tuple([2])
      else:
         ret = iden + key.serial
         if isinstance(key, Player):
            ret += tuple([0])
         elif isinstance(key, Tile):
            ret += tuple([1])

      pad = Serial.KEYLEN - len(ret)
      ret = tuple(pad*[-1]) + ret
      return ret
      #action: 2
      #tile: 1
      #player: 0

   def serialize(realm, ob, stim, outs):
      iden = realm.worldIdx, realm.tick

      #The environment is used to
      #generate serialization keys
      env, ent = ob
      key      = Serial.key(ent, iden)

      stim = stimulus.Dynamic.serialize(stim, iden)
      actn = action.Dynamic.serialize(outs, iden)

      return key, stim, actn

   def nontemporal(key):
      return tuple(key[0:1] + key[2:])

   def population(key):
      return key[2]

class RolloutManager:
   '''Manager class collecting and batching rollouts

   Notes:
      Includes inbuilt serialization with send/recv
      Access via self.rollouts[key].
   '''
   def __init__(self, postprocess=gammaDiscount):
      self.rollouts = {}
      self.done = {}
      self.log = []

      self.postprocess = postprocess
      self.nUpdates  = 0
      self.nRollouts = 0

   def __getitem__(self, key):
      if key not in self.rollouts:
         self.rollouts[key] = Rollout(self.postprocess)
      return self.rollouts[key]

   def logs(self):
      '''Returns log objects of all rollouts'''
      assert len(self.done) == 0
      self.nRollouts = 0
      logs = self.log
      self.log = []
      return logs

   '''
   def collectObservations(self, realm, obs, stims):
      #Collect rollouts on workers

      for ob, stim in zip(obs, stims):
         iden, stim, action = Serial.serialize(
            realm, ob, stim, action)

         self.nUpdates += 1
         key = Serial.nontemporal(iden)
         self[key].step(iden, stim, action, reward, done)
   '''

   def collect(self, realm, obs, stims, actions, rewards, dones):
      '''Collect rollouts on workers'''

      for ob, stim, action, reward, done in zip(
            obs, stims, actions, rewards, dones):

         iden, stim, action = Serial.serialize(
            realm, ob, stim, action)

         self.nUpdates += 1
         key = Serial.nontemporal(iden)
         self[key].step(iden, stim, action, reward, done)
 
   def send(self):
      '''Pack rollouts from workers for optim server'''
      keys, stims, actions, rewards, dones = [], [], [], [], []
      for key, rollout in self.rollouts.items():
         keys    += rollout.keys
         stims   += rollout.stims
         actions += rollout.actions
         rewards += rollout.rewards
         dones   += rollout.dones

      stims   = stimulus.Dynamic.batch(stims)
      actions = action.Dynamic.batch(actions)
      keys = np.stack(keys)

      return keys, stims, actions, rewards, dones

   def recv(self, packets):
      '''Unpack rollouts from workers on optim server
      
      Args: a list of serialized experience packets
      '''
      for sword, data in enumerate(packets):
         keys, stims, actions, rewards, dones = data

         stims   = stimulus.Dynamic.unbatch(stims)
         actions = action.Dynamic.unbatch(*actions)

         for iden, stim, atn, reward, done in zip(
               keys, stims, actions, rewards, dones):

            key = Serial.nontemporal(iden)
            self[key].step(iden, stim, atn, reward, done)

            if self[key].done:
               assert key not in self.done
               self.done[key] = self.rollouts[key]

               del self.rollouts[key]
               self.nRollouts += 1

   def fill(self, key, out, val, done):
      key = Serial.nontemporal(key)
      rollout = self.done[key]
      rollout.fill(key, out, val)

      if done:
         rollout.feather.finish()
         self.log.append(rollout.feather.blob)
         del self.done[key]

   def batched(self, batchSize):
      '''Returns flat batches of experience of the specified size

      Args:
         batchSize: batch size

      Notes:
         The last batch may be smaller than the specified sz
      '''
      return Batcher.batched(self.done, batchSize)


class Rollout:
   '''Rollout object

   Args:
      postprocess: reward trajectory post processing function
      serialzie: whether to serialize experience
   '''
   def __init__(self, postprocess):
      self.returnFunction = postprocess

      self.stims, self.actions = [], []
      self.rewards, self.dones = [], []
      self.outs, self.vals     = [], []
      self.keys                = []

      self.done = False
      self.time = 0

      #Logger
      self.feather = Feather()

   def __len__(self):
      assert self.time == len(self.stims)
      return self.time

   def fill(self, key, out, val):
      self.outs.append(out) 
      self.vals.append(val) 

      self.feather.scrawl(key)
      self.feather.value(val)

   #stim, action, reward
   #outs, val
   def step(self, iden, stim, atn, reward, done):
      '''Update rollout with one timestep of experience

      Args:
         key: The ID corresponding to the selected action
         out: the output logits of the network
         atn: the action chosen by the network
         val: the value function output of the network
         reward: the reward obtained from taking the chosen action
      '''
      self.stims.append(stim)
      self.actions.append(atn)
      self.rewards.append(reward)
      self.dones.append(done)

      #Unserialized keys
      self.keys.append(iden)

      self.time += 1
      self.done = done

      if done:
         self.finish()

      self.feather.reward(reward)

   def finish(self):
      '''Call once the full rollout has been collected'''
      assert self.rewards[-1] == -1
      self.returns = self.returnFunction(self.rewards)
      self.lifespan = len(self.rewards)

class Feather:
   '''Internal logger used by Rollout'''
   def __init__(self):
      self.expMap = set()
      self.blob = Blob()

   def scrawl(self, iden):
      '''Write logs from one time step

      Args:
         iden: The unique ID used in serialization
         atn: the actions chosen by the network
         val: the value function output of the network
         reward: the reward obtained from taking the chosen action
      '''
      world, annID, entID = iden
      self.blob.entID = entID
      self.blob.annID = annID
      self.blob.world = world
      
      #tile = self.tile(stim)
      #self.move(tile, ent.pos)
      #self.action(arguments, atnArgs)

   def tile(self, stim):
      R, C = stim.shape
      rCent, cCent = R//2, C//2
      tile = stim[rCent, cCent]
      return tile

   def action(self, arguments, atnArgs):
      move, attk = arguments
      moveArgs, attkArgs, _ = atnArgs
      moveLogits, moveIdx = moveArgs
      attkLogits, attkIdx = attkArgs

   def move(self, tile, pos):
      tile = type(tile.state)
      if pos not in self.expMap:
         self.expMap.add(pos)
         if tile in self.blob.unique:
            self.blob.unique[tile] += 1
      if tile in self.blob.counts:
         self.blob.counts[tile] += 1

   def reward(self, reward):
      self.blob.reward.append(reward)

   def value(self, value):
      self.blob.value.append(float(value))

   def finish(self):
      self.blob.finish()

