from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io import Stimulus as Static
from forge.ethyr.io import utils

class Data:
   '''Internal datastructure used in Stimulus processing'''
   def __init__(self):
      self.keys = []
      self.key = None
   
      self.data = defaultdict(list)

   def add(self, static, obj, *args, key):
      for name, attr in static:
         val = obj
         for n in name:
            val = getattr(val, n[0].lower() + n[1:])
         val = val.get(*args)
         if key != self.key:
            self.data[name].append([])
         self.data[name][-1].append(val)
      self.key = key
      self.keys.append(obj)

   @property
   def ret(self):
      return self.keys, self.data

class Stimulus:
   '''Static IO class used for interacting with game observations

   The environment returns game objects in observations; this class
   assembles them into usable data packets
   '''
   def process(env, ent, config, serialize):
      '''Main utility for processing game observations.

      Built to be semi-automatic and only require small updates
      for large new classes of observations

      Args:
         A list of observations as returned by the environment
      
      Returns:
         An unbatched list of proccessed stimulus packets
      '''
      keys      = 'Entity Tile'.split()
      functions = [Stimulus.entity, Stimulus.tile]

      ret = Stimulus.makeSets(env, ent, keys, functions)

      if serialize:
         ret = Stimulus.serialize(ret)

      return ret

   def basic(env, ent, config):
      N = env[config.STIM - 1, config.STIM].state.index == 0
      S = env[config.STIM + 1, config.STIM].state.index == 0
      W = env[config.STIM, config.STIM - 1].state.index == 0
      E = env[config.STIM, config.STIM + 1].state.index == 0
      return np.array([N, S, E, W]).astype(np.float)
      return [N, S, E, W]

   def makeSets(env, ent, keys, functions):
      '''Internal processor for sets of objects'''
      data, static = {}, Static.dict()
      for key, f in zip(keys, functions):
         data[key]  = f(env, ent, static[key])
      return data

   def serialize(stim):
      '''Internal stimulus serializer for communication across machines'''
      from forge.ethyr.io import Serial
      rets = {}
      for group, data in stim.items():
         names, data = data
         serialized = []
         for name in names:
            key = Serial.key(name)
            name.injectedSerial = key
            serialized.append(key)
         rets[group] = (serialized, data)
      return rets

   def batch(stims):
      '''Internal batcher for lists of stimuli'''
      #return np.array(stims)

      batch = {}

      #Process into set of sets
      for stim in stims:
         #Outer set
         for stat, stimSet in stim.items():
            if stat not in batch:
               batch[stat] = [[], defaultdict(list)]

            #Inner set
            keys, vals = stimSet
            batch[stat][0].append(keys)
            for attr, val in vals.items():
               val = np.array(val).reshape(-1)
               batch[stat][1][attr].append(val)

      #Pack values
      for group, stat in batch.items():
         keys, stat = stat
         for attr, vals in stat.items():
            vals, _ = utils.pack(vals)
            batch[group][1][attr] = vals

      return batch

   def unbatch(batch):
      '''Internal inverse batcher'''
      stims = []
      for group, stat in batch.items():
         keys, values = stat

         #Assign keys
         for idx, key in enumerate(keys):
            if idx == len(stims):
               stims.append(defaultdict(list))
            stims[idx][group] = [key, defaultdict(list)]

         #Assign values
         for attr, vals in values.items():
            lens = [len(e) for e in keys]
            vals = utils.unpack(vals, lens)
            for idx, val in enumerate(vals):
               stims[idx][group][1][attr] = val

      return stims
            
   def tile(env, ent, static):
      '''Internal processor for tile objects'''
      data = Data()
      for r, row in enumerate(env):
         for c, tile in enumerate(row):
            data.add(static, tile, tile, r, c, key=ent)

      '''
      popKeys = [('CRel',), ('RRel',), ('NEnts',)]
      for key in popKeys:
         data.data.pop(key)
      '''

      return data.ret

   def entity(env, ent, static):
      '''Internal processor for player objects. Always returns self first'''
      ents = []
      for tile in env.ravel():
         for e in tile.ents.values():
            ents.append(e)

      #Sorting to return self first.
      #This makes it easier to hack together cheap baselines
      data = Data()
      ents = sorted(ents, key=lambda e: e is ent, reverse=True)
      ents = ents[:1]
      while len(ents) < 1:
         ents.append(ent)

      for e in ents:
         data.add(static, e, ent, e, key=ent)

      '''
      popKeys = [('Status', 'Immune'), ('Status', 'Freeze'), ('History', 'TimeAlive'), ('History', 'Damage'), ('Base', 'C'), ('Base', 'R'), ('Base', 'Population')]
      for key in popKeys:
         data.data.pop(key)
      '''

      return data.ret
