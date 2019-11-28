from pdb import set_trace as T
from collections import defaultdict
import time

from forge.blade.io.action.static import Action as StaticAction

from forge.ethyr.io import Stimulus, Action, Serial, utils
from forge.ethyr.io.action import ActionArgs

class Lookup:
   '''Lookup utility for indexing 
   (name, data) pairs'''
   def __init__(self):
      self.data = {}
      self.back = {}
      self.max = 0

   def add(self, name, idx=None, orig=None):
      '''Add entries to the table'''
      if idx is None:
         idx = self.max

      assert name not in self.data
      self.data[name] = idx
      if orig is not None:
         self.back[idx] = orig
      self.max += 1

      return idx

   def __contains__(self, key):
      return key in self.data

   def reverse(self, idx):
      return self.back[idx]

class Output:
   def __init__(self, key, atn, atnIdx, value):
      self.key    = key
      self.out    = (key, atnIdx, atn)

      self.action = {key: atnIdx}
      self.value  = value

class ObsData:
   def __init__(self):
      self.entities = defaultdict(EntityData)
      self.names = defaultdict(list)
      self.n = 0

class EntityData:
   def __init__(self):
      self.attributes = defaultdict(list)

class AtnData:
   def __init__(self):
      self.actions = defaultdict(ArgsData)
      self.names = []

class ArgsData:
   def __init__(self):
      self.arguments = defaultdict(list)

class Inp:
   def __init__(self):
      self.obs    = ObsData()
      self.atn    = AtnData()
      self.lookup  = Lookup()

      self.keys    = []
      self.rewards = []
      self.dones   = []

   def actions(self, serialize=True):
      for atn in StaticAction.arguments:
         serial = atn
         if serialize:
            serial = Serial.key(serial)
         self.lookup.add(serial, orig=atn)

      #Set the 0 index embedding to pad
      zero_key = (0, 0) + tuple([0]*Serial.KEYLEN)
      negs_key = (0, 0) + tuple([-1]*Serial.KEYLEN)

      self.lookup.add(zero_key, orig='ZERO_PAD')
      #self.lookup.add(zero_key, orig='NULL_PAD')

   def key(self, env, ent, reward, config):

      annID, entID = ent.annID, ent.entID
      key = (annID, entID)
      self.keys.append(key)
      self.rewards.append(reward)

   def pack(self):
      for atn, action in self.atn.actions.items():
         for arg, argument in action.arguments.items():
            tensor, lens = utils.pack(argument)
            self.atn.actions[atn].arguments[arg] = tuple([tensor, lens])
  
class IO:
   '''High level I/O class for abstracting game state and action selection'''
   def inputs(obs, rewards, dones, groupFn, config, serialize):
      '''Preprocess inputs'''
      inputs = defaultdict(Inp)
      for done in dones:
         idx = groupFn(done[1])
         inputs[idx].dones.append(done)

      ### Process inputs
      for ob, reward in zip(obs, rewards):
         env, ent = ob
         idx = groupFn(ent.entID)
         inputs[idx].key(env, ent, reward, config)
         Stimulus.process(inputs[idx], env, ent, config, serialize)
      
      start = time.time()
      #Index actions
      for idx, inp in inputs.items():
         inputs[idx].actions()

      ### Process outputs
      for ob, reward in zip(obs, rewards):
         env, ent = ob
         idx = groupFn(ent.entID)
         Action.process(inputs[idx], env, ent, config, serialize)
   
      #Pack actions
      for idx, inp in inputs.items():
         inputs[idx].pack()

      return inputs

   def outputs(obs, atnDict=None):
      '''Postprocess outputs'''

      #Initialize output dictionary
      if atnDict is None:
         atnDict = defaultdict(lambda: defaultdict(list))

      #Reverse format lookup over actions
      for atn, action in obs.atn.actions.items():
         for arg, atnsIdx in action.arguments.items():
            for idx, a in enumerate(atnsIdx[1]):
               names = list(obs.obs.names.keys())
               _, entID, _= names[idx]

               a = obs.lookup.reverse(a)
               atnDict[entID][atn].append(a)

      return atnDict 
