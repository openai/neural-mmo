from pdb import set_trace as T
import numpy as np

from itertools import chain 
from collections import defaultdict

from forge.blade.io.action import Node
from forge.ethyr.io import Stimulus, Action

class Serial:
   KEYLEN = 5
   ACTION = 2
   TILE   = 1
   PLAYER = 0

   '''Internal serialization class for communication across machines

   Mainly wraps Stimulus.serialize and Action.serialize. Also provides
   keying functionality for converting game objects to unique IDs.'''
   def key(key, iden):
      '''Convert a game object to a unique key'''
      from forge.blade.entity import Player
      from forge.blade.core.tile import Tile

      ret = key.serial
      if isinstance(key, type):
         if issubclass(key, Node):
            ret += tuple([Serial.ACTION])
      else:
         ret = iden + key.serial
         if isinstance(key, Player):
            ret += tuple([Serial.PLAYER])
         elif isinstance(key, Tile):
            ret += tuple([Serial.TILE])

      pad = Serial.KEYLEN - len(ret)
      ret = tuple(pad*[-1]) + ret
      return ret

   def realmKey(realm, ob):
      iden = realm.worldIdx, realm.tick

      #The environment is used to
      #generate serialization keys
      env, ent = ob
      key      = Serial.key(ent, iden)
      return iden, key

   def inputs(realm, ob, stim):
      '''Serialize observations'''
      iden, key = Serial.realmKey(realm, ob)
      stim = Stimulus.serialize(stim, iden)

      return iden, key, stim

   def outputs(realm, ob, outs):
      '''Serialize actions'''
      iden, key = Serial.realmKey(realm, ob)
      actn = Action.serialize(outs, iden)
      return iden, key, actn

   def nontemporal(key):
      '''Get the time independent part of a key'''
      return tuple(key[:1]) + tuple(key[2:])

   def population(key):
      '''Get the population component of a nontemporal entity key'''
      return key[1]

