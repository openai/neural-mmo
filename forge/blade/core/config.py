from pdb import set_trace as T
import numpy as np
import inspect
import os

from collections import defaultdict
from itertools import chain

class StaticIterable(type):
   def __iter__(cls):
      stack = list(cls.__dict__.items())
      for name, attr in cls.__dict__.items():
         if name == '__module__':
            continue
         if name.startswith('__'):
            break
         yield name, attr

class Template(metaclass=StaticIterable):
   def __init__(self):
      self.data = {}
      cls       = type(self)

      #Set defaults from static properties
      for k, v in cls:
         self.set(k, v)

   def override(self, **kwargs):
      for k, v in kwargs.items():
         err = 'CLI argument: {} is not a Config property'.format(k)
         assert hasattr(self, k), err
         self.set(k, v)

   def set(self, k, v):
      setattr(self, k, v)
      self.data[k] = v

   def print(self):
      keyLen = 0
      for k in self.data.keys():
         keyLen = max(keyLen, len(k))

      print('Configuration')
      for k, v in self.data.items():
         print('   {:{}s}: {}'.format(k, keyLen, v))

class Config(Template):
   '''An environment configuration object

   Global constants are defined as static class variables. You can override
   any Config variable using standard CLI syntax (e.g. --NENT=128). 
   
   Notes:
      We use Google Fire internally to replace standard manual argparse
      definitions for each Config property. This means you can subclass
      Config to add new static attributes -- CLI definitions will be
      generated automatically.
   '''
   ROOT   = os.path.join(os.getcwd(), 'resource/maps/procedural/map')
   SUFFIX = '/map.tmx'

   NTILE  = 6 #Number of distinct tile types
   SZ     = 62
   BORDER = 9
   R = C  = SZ + BORDER

   #Agent name
   NAME_PREFIX             = 'Neural_'
   '''Prefix used in agent names displayed by the client'''


   # Observation tile crop size
   STIM                    = 7
   '''Number of tiles an agent can see in any direction'''

   WINDOW                  = 2*STIM + 1
   '''Size of the square tile crop visible to an agent'''


   # Population parameters
   NENT                    = 256
   '''Maximum number of agents spawnable in the environment'''

   NPOP                    = 8
   '''Number of distinct populations spawnable in the environment'''


   # Skill parameters
   RESOURCE                = 10
   '''Initial level and capacity for Hunting + Fishing resource skills'''

   HEALTH                  = 10
   '''Initial Constitution level and agent health'''

   HEALTH_REGEN_THRESHOLD  = 0.5
   '''Fraction of maximum resource capacity required to regen health'''

   RESOURCE_RESTORE        = 0.5
   '''Fraction of maximum capacity restored upon collecting a resource'''

   HEALTH_RESTORE          = 0.1
   '''Fraction of health restored per tick when above half food+water'''


   # Experience parameters
   XP_SCALE                = 10
   '''Skill level progression speed as a multiplier of typical MMOs'''

   CONSTITUTION_XP_SCALE   = 2
   '''Multiplier on top of XP_SCALE for the Constitution skill'''

   COMBAT_XP_SCALE         = 4
   '''Multiplier on top of XP_SCALE for Combat skills'''


   # Combat parameters
   IMMUNE                  = 10
   '''Number of ticks an agent cannot be damaged after spawning'''

   MELEE_RANGE             = 1
   '''Range of attacks using the Melee skill'''

   RANGE_RANGE             = 3
   '''Range of attacks using the Range skill'''

   MAGE_RANGE              = 4
   '''Range of attacks using the Mage skill'''

   FREEZE_TIME             = 3
   '''Number of ticks successful Mage attacks freeze a target'''

   def SPAWN(self):
      '''Generates spawn positions for new agents

      Default behavior randomly selects a tile position
      along the borders of the square game map

      Returns:
         tuple(int, int):

         position:
            The position (row, col) to spawn the given agent
      '''
      R, C = Config.R, Config.C
      spawn, border, sz = [], Config.BORDER, Config.SZ
      spawn += [(border, border+i) for i in range(sz)]
      spawn += [(border+i, border) for i in range(sz)]
      spawn += [(R-1, border+i) for i in range(sz)]
      spawn += [(border+i, C-1) for i in range(sz)]
      idx = np.random.randint(0, len(spawn))
      return spawn[idx]
