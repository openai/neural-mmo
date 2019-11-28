from pdb import set_trace as T
import numpy as np

from itertools import chain 
from collections import defaultdict

from forge.blade.io.action import Node
#from forge.ethyr.io import Stimulus, Action

class Serial:
   KEYLEN = 3
   ACTION = 2
   TILE   = 1
   PLAYER = 0

   '''Internal serialization class for communication across machines

   Mainly wraps Stimulus.serialize and Action.serialize. Also provides
   keying functionality for converting game objects to unique IDs.

   Format: World, Tick, key.serial[0], key.serial[1], key type'''
   def key(key):
      '''Convert a game object to a unique key'''
      from forge.blade.entity import Player
      from forge.blade.core.tile import Tile

      if key is None:
         return tuple([-1]*Serial.KEYLEN)

      ret = key.serial
      if isinstance(key, type):
         if issubclass(key, Node):
            ret += tuple([Serial.ACTION])
      else:
         ret = key.serial
         if isinstance(key, Player):
            ret += tuple([Serial.PLAYER])
         elif isinstance(key, Tile):
            ret += tuple([Serial.TILE])

      pad = Serial.KEYLEN - len(ret)
      ret = tuple(pad*[-1]) + ret
      return ret

   #def inputs(realm, stim):
   #   '''Serialize observations'''
   #   iden, key = Serial.realmKey(realm, stim)
   #   stim = Stimulus.serialize(stim, iden)
   #   return iden, key, stim

   #def outputs(realm, ob, outs):
   #   '''Serialize actions'''
   #   #Offset environment tick by 1 because we have
   #   #stepped the environment.
   #   iden, key = Serial.realmKey(realm, ob, 1)
   #   actn = Action.serialize(outs, iden)
   #   return iden, key, actn

   def nontemporal(key):
      '''Get the time independent part of a key'''
      return tuple(key[:1]) + tuple(key[2:])

   def population(key):
      '''Get the population component of a nontemporal entity key'''
      return key[1]

