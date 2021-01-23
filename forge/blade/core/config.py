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
   ############################################################################
   ### Meta-Parameters
   ENV_NAME             = 'Neural_MMO'
   '''Environment Name'''

   ENV_VERSION          = '1.5'
   '''Environment version'''

   EVALUATE             = False
   '''Run in evaluation mode (test/render)'''

   v                    = False
   '''Verbose mode'''


   ############################################################################
   ### Path Parameters
   PATH_ROOT            = os.getcwd()
   '''Global repository directory'''

   PATH_RESOURCE        = os.path.join(PATH_ROOT, 'resource')
   '''Resource directory'''


   #Maps
   PATH_MAPS            = os.path.join(PATH_RESOURCE, 'maps')
   '''Generated map directory'''

   PATH_MAP_SUFFIX      = 'map{}/map.npy'
   '''Map file name'''

   PATH_MAPS_SMALL      = os.path.join(PATH_MAPS, 'procedural-small')
   '''Generated map directory for SmallMap config'''

   PATH_MAPS_LARGE      = os.path.join(PATH_MAPS, 'procedural-large')
   '''Generated map directory for LargeMap config'''


   #Assets
   PATH_ASSETS          = os.path.join(PATH_RESOURCE, 'assets')
   '''Asset directory'''

   PATH_FONTS           = os.path.join(PATH_ASSETS, 'fonts')
   '''Font directory'''

   PATH_LOGO            = os.path.join(PATH_FONTS, 'ascii.txt')
   '''Logo file (Ascii art)'''


   #Logs
   PATH_LOGS            = 'experiment'
   '''Training and evaluation log directory'''

   PATH_EVAL_DATA       = os.path.join(PATH_LOGS, 'evaluation.npy')
   '''Evaluation data file'''

   PATH_EVAL_FIGURE     = os.path.join(PATH_LOGS, 'evaluation.html')
   '''Evaluation figure file'''

   #Themes
   PATH_THEMES          = os.path.join('forge', 'blade', 'systems', 'visualizer') 
   '''Theme directory'''

   PATH_THEME_WEB       = os.path.join(PATH_THEMES, 'index_web.html')
   '''Web theme file'''

   PATH_THEME_PUB       = os.path.join(PATH_THEMES, 'index_publication.html')
   '''Publication theme file'''


   ############################################################################
   ### Visualization Parameters
   VIS_THEME            = 'web'
   '''Visualizer theme: web or publication'''

   VIS_WIDTH            = 1920
   '''Visualizer figure width (pixels)'''

   VIS_HEIGHT           = 314
   '''Visualizer per-plot height (pixels)'''

   VIS_BORDER_WIDTH     = 20
   '''Horizontal padding per figure side (pixels)'''

   VIS_BORDER_HEIGHT    = 60
   '''Vertical padding per figure side (pixels)'''

   VIS_LEGEND_WIDTH     = 109
   '''Width of legend label before offset'''
   
   VIS_LEGEND_OFFSET    = 71 
   '''Width of legend label offset'''

   VIS_TITLE_OFFSET     = 60
   '''Width of left title offset'''

   VIS_PORT             = 5006
   '''Visualizer local Bokeh server port'''

   VIS_TOOLS            = False
   '''Visualizer display plot tools'''

   ############################################################################
   ### Terrain Generation Parameters
   TERRAIN_RENDER       = False
   '''Whether map generation should also save .png previews (slow + large file size)'''

   TERRAIN_SIZE         = 1024
   '''Size of each map (number of tiles along each side)'''

   TERRAIN_BORDER       = 10
   '''Number of lava border tiles surrounding each side of the map'''

   TERRAIN_FREQUENCY    = (-3, -6)
   '''Simplex noise frequence range (log2 space)'''

   TERRAIN_OCTAVES      = 8
   '''Number of octaves sampled from log2 spaced TERRAIN_FREQUENCY range'''

   TERRAIN_MODE         = 'expand'
   '''expand or contract.

   Specify normal generation (lower frequency at map center) or
   inverted generation (lower frequency at map edges)'''

   TERRAIN_LERP         = True 
   '''Whether to apply a linear blend between terrain octaves'''

   TERRAIN_ALPHA        = 0.15
   '''Blend factor for FOREST_LOW (water adjacent)'''

   TERRAIN_BETA         = 0.025
   '''Blend factor for FOREST_HIGH (stone adjacent)'''

   TERRAIN_LAVA         = 0.0
   '''Noise threshold for lava generation'''

   TERRAIN_WATER        = 0.25
   '''Noise threshold for water generation'''

   TERRAIN_FOREST_LOW   = 0.35
   '''Noise threshold for forest (water adjacent)'''

   TERRAIN_GRASS        = 0.75
   '''Noise threshold for grass'''

   TERRAIN_FOREST_HIGH  = 0.775
   '''Noise threshold for forest (stone adjacent)'''

   TERRAIN_WATER_RADIUS  = 3.5
   '''Central water radius'''

   TERRAIN_CENTER_REGION = 19 #Keep this number odd for large maps
   '''Central water square cutout'''

   TERRAIN_CENTER_WIDTH  = 3
   '''Central square grass border'''


   ############################################################################
   ### Tile Parameters
   FOREST_CAPACITY      = 1
   '''Maximum number of harvests before a forest tile decays'''

   FOREST_RESPAWN       = 0.025
   '''Probability that a harvested forest tile will regenerate each tick'''

   OREROCK_CAPACITY     = 1
   '''Maximum number of harvests before an orerock tile decays'''

   OREROCK_RESPAWN      = 0.05
   '''Probability that a harvested orerock tile will regenerate each tick'''
 
   NTILE  = 6
   '''Number of distinct terrain tile types'''


   ############################################################################
   ### Population Parameters                                                   
   NENT                    = 256
   '''Maximum number of agents spawnable in the environment'''

   NMOB                    = 1024
   '''Maximum number of NPCs spawnable in the environment'''

   NPOP                    = 1
   '''Number of distinct populations spawnable in the environment'''

   NMAPS                   = 256 #Number of maps to generate


   ############################################################################
   ### Agent Parameters
   NAME_PREFIX             = 'Neural_'
   '''Prefix used in agent names displayed by the client'''

   STIM                    = 7
   '''Number of tiles an agent can see in any direction'''


   ############################################################################
   ### Experience Parameters                                                   
   XP_SCALE                = 10
   '''Skill level progression speed as a multiplier of typical MMOs'''

   CONSTITUTION_XP_SCALE   = 2
   '''Multiplier on top of XP_SCALE for the Constitution skill'''

   COMBAT_XP_SCALE         = 4
   '''Multiplier on top of XP_SCALE for Combat skills'''


   ############################################################################
   ### Skill Parameters                                                   
   RESOURCE                = 10
   '''Initial level and capacity for Hunting + Fishing resource skills'''

   HEALTH                  = 10
   '''Initial Constitution level and agent health'''

   HEALTH_REGEN_THRESHOLD  = 0.5
   '''Fraction of maximum resource capacity required to regen health'''

   RESOURCE_RESTORE        = 1.0 #Modified from .5... Small maps?
   '''Fraction of maximum capacity restored upon collecting a resource'''

   HEALTH_RESTORE          = 0.1
   '''Fraction of health restored per tick when above half food+water'''

   DEFENSE_WEIGHT          = 0.3 
   '''Fraction of defense that comes from the Defense skill'''   

   DICE_SIDES              = 20
   '''Number of sides for combat dice

   Attacks can only hit opponents up to the attacker's level plus
   DICE_SIDES/2. Increasing this value makes attacks more accurate
   and allows lower level attackers to hit stronger opponents'''

   MELEE_RANGE             = 1
   '''Range of attacks using the Melee skill'''

   RANGE_RANGE             = 3
   '''Range of attacks using the Range skill'''

   MAGE_RANGE              = 4
   '''Range of attacks using the Mage skill'''

   FREEZE_TIME             = 3
   '''Number of ticks successful Mage attacks freeze a target'''


   ############################################################################
   ### Spawn Protection Parameters                                             
   IMMUNE_ADD              = 10
   '''Minimum number of ticks an agent cannot be damaged after spawning'''

   IMMUNE_MUL              = 0.05
   '''Additional number of immunity ticks per population size'''

   IMMUNE_MAX              = 50
   '''Maximum number of immunity ticks'''

   WILDERNESS              = True
   '''Whether to bracket terrain into combat level ranges'''

   INVERT_WILDERNESS       = False
   '''Whether to invert wilderness level generation'''

   WILDERNESS_MIN          = -1
   '''Minimum wilderness level. -1 corresponds to a safe zone'''

   WILDERNESS_MAX          = 99
   '''Maximum wilderness level. 99 corresponds to unrestricted combat'''


   ############################################################################
   ### Spawn Parameters                                                   
   PLAYER_SPAWN_ATTEMPTS   = 3
   '''Number of player spawn attempts per tick

   Note that the env will attempt to spawn agents until success
   if the current population size is zero.'''

   NPC_SPAWN_ATTEMPTS      = 25
   '''Number of NPC spawn attempts per tick'''

   SPAWN_CENTER            = True
   '''Whether to spawn agents from the map center or edges'''

   NPC_SPAWN_AGGRESSIVE    = 0.75
   '''Percentage distance threshold from spawn for aggressive NPCs'''

   NPC_SPAWN_NEUTRAL       = 0.40
   '''Percentage distance threshold from spawn for neutral NPCs'''

   NPC_SPAWN_PASSIVE       = 0.02
   '''Percentage distance threshold from spawn for passive NPCs'''
   
   NPC_LEVEL_MIN           = 1
   '''Minimum NPC level'''

   NPC_LEVEL_MAX           = 99
   '''Maximum NPC level'''

   NPC_LEVEL_SPREAD        = 10 
   '''Level range for NPC spawns'''

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
