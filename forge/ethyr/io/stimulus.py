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
         val = getattr(obj, attr.name).get(*args)
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
   def process(stims):
      '''Main utility for processing game observations.

      Built to be semi-automatic and only require small updates
      for large new classes of observations

      Args:
         A list of observations as returned by the environment
      
      Returns:
         An unbatched list of proccessed stimulus packets
      '''
      rets = [] 
      for stim in stims:
         keys      = 'Entity Tile'.split()
         functions = [Stimulus.entity, Stimulus.tile]
         ret = Stimulus.makeSets(stim, keys, functions)
         rets.append(ret)
      return rets

   def makeSets(stim, keys, functions):
      '''Internal processor for sets of objects'''
      env, ent = stim
      data, static = {}, dict(Static)
      for key, f in zip(keys, functions):
         data[key]  = f(env, ent, static[key])
      return data

   def serialize(stim, iden):
      '''Internal stimulus serializer for communication across machines'''
      from forge.ethyr.io import Serial
      rets = {}
      for group, data in stim.items():
         names, data = data
         serialNames = []
         for name in names:
            serialName = Serial.key(name, iden)
            name.injectedSerial = serialName
            serialNames.append(serialName)
         rets[group] = (serialNames, data)
      return rets

   def batch(stims):
      '''Internal batcher for lists of stimuli'''
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
      return data.ret

   def entity(env, ent, static):
      '''Internal processor for player objects'''
      data = Data()
      for tile in env.ravel():
         for e in tile.ents.values():
            data.add(static, e, ent, e, key=ent)
      return data.ret
