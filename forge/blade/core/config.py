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

   TERRAIN_DIR_SMALL = 'resource/maps/procedural-small'
   TERRAIN_DIR_LARGE = 'resource/maps/procedural-large'

   ASSETS_DIR        = 'resource/assets'

   LOGO_FILE         = 'fonts/ascii.txt'
   LOGO_DIR          =  os.path.join(ASSETS_DIR, LOGO_FILE)

   #Terrain parameters
   TERRAIN_DIR       = TERRAIN_DIR_LARGE
   '''Directory in which generated maps are saved'''

   TERRAIN_RENDER    = False
   '''Whether map generation should also save .png previews (slow + large file size)'''

   TERRAIN_SIZE      = 1024
   '''Size of each map (number of tiles along each side)'''

   TERRAIN_BORDER    = 10
   '''Number of lava border tiles surrounding each side of the map'''

   TERRAIN_FREQUENCY = (-3, -6)
   '''Simplex noise frequence range (log2 space)'''

   TERRAIN_OCTAVES   = 8
   '''Number of octaves sampled from log2 spaced TERRAIN_FREQUENCY range'''

   TERRAIN_MODE      = 'expand'
   '''expand or contract. Specify normal generation (lower frequency at map center) or inverted generation (lower frequency at map edges)'''

   TERRAIN_LERP      = True 

   TERRAIN_ALPHA        = 0.15
   TERRAIN_BETA         = 0.025
   TERRAIN_LAVA         = 0.0
   TERRAIN_WATER        = 0.25
   TERRAIN_FOREST_LOW   = 0.35
   TERRAIN_GRASS        = 0.75
   TERRAIN_FOREST_HIGH  = 0.775

   TERRAIN_WATER_RADIUS  = 3.5
   TERRAIN_CENTER_REGION = 19 #Keep this number odd for large maps
   TERRAIN_CENTER_WIDTH  = 3

   #Map load parameters
   ROOT   = os.path.join(os.getcwd(), TERRAIN_DIR, 'map')
   '''Terrain map load directory'''

   SUFFIX = '/map.npy'
   '''Terrain map file suffix'''

   NTILE  = 6
   '''Number of distinct tile types'''

   #Agent name
   NAME_PREFIX             = 'Neural_'
   '''Prefix used in agent names displayed by the client'''


   # Observation tile crop size
   STIM                    = 7
   '''Number of tiles an agent can see in any direction'''


   # Population parameters
   NENT                    = 256
   '''Maximum number of agents spawnable in the environment'''

   NMOB                    = 1024
   '''Maximum number of NPCs spawnable in the environment'''

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
   IMMUNE                  = 0
   '''Number of ticks an agent cannot be damaged after spawning'''

   WILDERNESS              = True
   '''Whether to bracket combat into wilderness levels'''

   INVERT_WILDERNESS       = False
   '''Whether to reverse wilderness level generation'''

   WILDERNESS_MIN          = -1
   WILDERNESS_MAX          = 99

   #Fix this to be for small maps in systems skill
   RESOURCE_GRACE_PERIOD   = -1 

   PLAYER_SPAWN_ATTEMPTS   = 5
   NPC_SPAWN_ATTEMPTS      = 25

   NPC_SPAWN_AGGRESSIVE    = 0.75
   NPC_SPAWN_NEUTRAL       = 0.40
   NPC_SPAWN_PASSIVE       = 0.02
   
   NPC_LEVEL_MIN           = 1
   NPC_LEVEL_MAX           = 99
   NPC_LEVEL_SPREAD        = 10 


   DEFENSE_WEIGHT          = 0.3 
   '''Fraction of defense that comes from the Defense skill'''   

   DICE_SIDES              = 20
   '''Number of sides for combat dice -- higher means weaker attacks can hit stronger opponents'''

   MELEE_RANGE             = 1
   '''Range of attacks using the Melee skill'''

   RANGE_RANGE             = 3
   '''Range of attacks using the Range skill'''

   MAGE_RANGE              = 4
   '''Range of attacks using the Mage skill'''

   FREEZE_TIME             = 3
   '''Number of ticks successful Mage attacks freeze a target'''

   @property
   def WINDOW(self):
      '''Size of the square tile crop visible to an agent'''
      return 2*self.STIM + 1

   def SPAWN(self):
      '''Generates spawn positions for new agents

      Default behavior randomly selects a tile position
      along the borders of the square game map

      Returns:
         tuple(int, int):

         position:
            The position (row, col) to spawn the given agent
      '''
      #Spawn at edges
      if self.TERRAIN_MODE == 'contract':
         mmax = self.TERRAIN_SIZE - self.TERRAIN_BORDER - 1
         mmin = self.TERRAIN_BORDER

         var  = np.random.randint(mmin, mmax)
         fixed = np.random.choice([mmin, mmax])
         r, c = int(var), int(fixed)
         if np.random.rand() > 0.5:
             r, c = c, r 
         return (r, c)
      #Spawn at center
      else:
         spawnRadius = self.TERRAIN_CENTER_REGION
         spawnWidth  = self.TERRAIN_CENTER_WIDTH

         cent  = self.TERRAIN_SIZE // 2
         left  = cent - self.TERRAIN_CENTER_REGION
         right = cent + self.TERRAIN_CENTER_REGION

         var  = np.random.randint(left, right)
         if np.random.rand() > 0.5:
            fixed = np.random.randint(left, left+spawnWidth)
         else:
            fixed = np.random.randint(right-spawnWidth, right)

         r, c = int(var), int(fixed)
         if np.random.rand() > 0.5:
            r, c = c, r
         return r, c
