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

   ENV_VERSION          = '1.5.1'
   '''Environment version'''

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

   PATH_TILE            = os.path.join(PATH_ASSETS, 'tiles/{}.png')
   '''Tile path -- format me with tile name'''

   PATH_TEXT            = os.path.join(PATH_ASSETS, 'text')
   '''Text directory'''

   PATH_LOGO            = os.path.join(PATH_TEXT, 'ascii.txt')
   '''Logo file (Ascii art)'''


   #Baselines and Checkpoints
   PATH_CHECKPOINTS          = 'checkpoints'
   '''Checkpoints path'''

   PATH_BASELINES            = 'baselines'
   '''Model and evaluation directory'''
  
   PATH_ALL_MODELS           = os.path.join(PATH_BASELINES, 'models')
   '''All models directory'''

   @property
   def PATH_MODEL(self):
      '''Model path'''
      return os.path.join(self.PATH_ALL_MODELS, self.MODEL)

   @property
   def PATH_TRAINING_DATA(self):
      '''Model training data'''
      return os.path.join(self.PATH_MODEL, 'training.npy')

   PATH_ALL_EVALUATIONS      = os.path.join(PATH_BASELINES, 'evaluations')
   '''All evaluations directory'''

   @property
   def PATH_EVALUATION(self):
      '''Evaluation path'''
      return os.path.join(self.PATH_ALL_EVALUATIONS, self.NAME, self.MODEL, '.npy')

   #Themes
   PATH_THEMES          = os.path.join('forge', 'blade', 'systems', 'visualizer') 
   '''Theme directory'''

   PATH_THEME_WEB       = os.path.join(PATH_THEMES, 'index_web.html')
   '''Web theme file'''

   PATH_THEME_PUB       = os.path.join(PATH_THEMES, 'index_publication.html')
   '''Publication theme file'''

   ############################################################################
   ### Evaluation Parameters
   EVALUATE             = False
   '''Flag used by evaluation mode'''

   RENDER               = False
   '''Flag used by rener mode'''

   EVAL_MAPS            = 5
   '''Number of evaluation maps'''

   GENERALIZE           = True
   '''Evaluate on maps not seen during training'''

   TRAIN_SUMMARY_ENVS   = 10
   '''Most recent envs to use for training summaries'''

   TRAIN_DATA_RESAMPLE  = 200
   '''Number of points to resample training data'''


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
   TERRAIN_RENDER             = False
   '''Whether map generation should also save .png previews (slow + large file size)'''

   TERRAIN_CENTER             = 1024
   '''Size of each map (number of tiles along each side)'''

   TERRAIN_BORDER             = 10
   '''Number of lava border tiles surrounding each side of the map'''

   @property
   def TERRAIN_SIZE(self):
      return int(self.TERRAIN_CENTER + 2*self.TERRAIN_BORDER)

   TERRAIN_FREQUENCY          = -3
   '''Base noise frequency range (log2 space)'''

   TERRAIN_FREQUENCY_OFFSET   = 7
   '''Noise frequency octave offset (log2 space)'''

   TERRAIN_LOG_INTERPOLATE_MIN = -2 
   '''Minimum interpolation log-strength for noise frequencies'''

   TERRAIN_LOG_INTERPOLATE_MAX= 0
   '''Maximum interpolation log-strength for noise frequencies'''

   TERRAIN_TILES_PER_OCTAVE   = 8
   '''Number of octaves sampled from log2 spaced TERRAIN_FREQUENCY range'''

   TERRAIN_LAVA               = 0.0
   '''Noise threshold for lava generation'''

   TERRAIN_WATER              = 0.30
   '''Noise threshold for water generation'''

   TERRAIN_GRASS              = 0.70
   '''Noise threshold for grass'''

   TERRAIN_FOREST             = 0.85
   '''Noise threshold for forest'''

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

   N_TRAIN_MAPS            = 256
   '''Number of training maps to generate'''

   N_EVAL_MAPS             = 64
   '''Number of evaluation maps to generate'''

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
      mmax = self.TERRAIN_CENTER + self.TERRAIN_BORDER
      mmin = self.TERRAIN_BORDER

      var  = np.random.randint(mmin, mmax)
      fixed = np.random.choice([mmin, mmax])
      r, c = int(var), int(fixed)
      if np.random.rand() > 0.5:
          r, c = c, r 
      return (r, c)
