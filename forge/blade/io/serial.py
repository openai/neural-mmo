from pdb import set_trace as T
import numpy as np

from itertools import chain 
from collections import defaultdict

class Serial:
   '''Internal serialization class for communication across machines

   Mainly wraps Stimulus.serialize and Action.serialize. Also provides
   keying functionality for converting game objects to unique IDs.

   Format: World, Tick, key.serial[0], key.serial[1], key type'''

   #Length of serial key tuple
   KEYLEN = 3

   def key(key):
      '''Convert a game object to a unique key'''
      if key is None:
         return tuple([-1]*Serial.KEYLEN)

      #Concat object key with class key
      ret = key.serial + tuple([key.SERIAL])
      pad = Serial.KEYLEN - len(ret)
      ret = tuple(pad*[-1]) + ret
      return ret

   def nontemporal(key):
      '''Get the time independent part of a key'''
      return tuple(key[:1]) + tuple(key[2:])

   def population(key):
      '''Get the population component of a nontemporal entity key'''
      return key[1]

