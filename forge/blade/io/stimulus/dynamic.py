from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io.stimulus import Static
from forge.blade.io import utils

class Data:
   def __init__(self):
      self.keys = []
      self.key = None
   
      self.data = defaultdict(list)

   def add(self, static, obj, *args, key):
      for name, attr in static:
         val = getattr(obj, attr.name).get(*args)
         if key != self.key:
            self.data[name].append([])
         self.data[name][-1].append(val)
      self.key = key
      self.keys.append(obj)

   @property
   def ret(self):
      return self.keys, self.data

class Dynamic:
   def __call__(self, stim):
      keys      = 'Entity Tile'.split()
      functions = [self.entity, self.tile]
      return self.makeSets(stim, keys, functions)

   def makeSets(self, stim, keys, functions):
      env, ent = stim
      data, static = {}, dict(Static)
      for key, f in zip(keys, functions):
         data[key]  = f(env, ent, static[key])
      return data

   def serialize(stim, iden):
      rets = {}
      for group, data in stim.items():
         names, data = data
         names = [(iden + e.serial) for e in names]
         rets[group] = (names, data)
      return rets

   def batch(stims):
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
            
   def tile(self, env, ent, static):
      data = Data()
      env = env[6:9, 6:9]
      for r, row in enumerate(env):
         for c, tile in enumerate(row):
            data.add(static, tile, tile, r, c, key=ent)
      return data.ret

   def entity(self, env, ent, static):
      data = Data()
      data.add(static, ent, ent, ent, key=ent)
      '''
      for tile in env.ravel():
         for e in tile.ents.values():
            data.add(static, e, ent, e, key=ent)
      '''
      return data.ret
