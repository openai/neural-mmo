from pdb import set_trace as T
import numpy as np
import os

import nmmo
from nmmo.lib import utils

class SequentialLoader:
    '''config.AGENT_LOADER that spreads out agent populations'''
    def __init__(self, config):
        items = config.AGENTS
        for idx, itm in enumerate(items):
           itm.policyID = idx 

        self.items = items
        self.idx   = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.idx = (self.idx + 1) % len(self.items)
        return self.idx, self.items[self.idx]

class TeamLoader:
    '''config.AGENT_LOADER that loads agent populations adjacent'''
    def __init__(self, config):
        items = config.AGENTS
        self.team_size = config.NENT // config.NPOP

        for idx, itm in enumerate(items):
           itm.policyID = idx 

        self.items = items
        self.idx   = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        team_idx  = self.idx // self.team_size
        return team_idx, self.items[team_idx]


class Template(metaclass=utils.StaticIterable):
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
      if type(v) is not property:
         try:
            setattr(self, k, v)
         except:
            print('Cannot set attribute: {} to {}'.format(k, v))
            quit()
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

   The default config as of v1.5 uses 1024x1024 maps with up to 2048 agents
   and 1024 NPCs. It is suitable to time horizons of 8192+ steps. For smaller
   experiments, consider the SmallMaps config.
   
   Notes:
      We use Google Fire internally to replace standard manual argparse
      definitions for each Config property. This means you can subclass
      Config to add new static attributes -- CLI definitions will be
      generated automatically.
   '''

   def __init__(self):
      super().__init__()

      if __debug__:
         err = 'config.Config is a base class. Use config.{Small, Medium Large}'''
         assert type(self) != Config, err

   ############################################################################
   ### Meta-Parameters
   RENDER                 = False
   '''Flag used by render mode'''

   def game_system_enabled(self, name) -> bool:
      return hasattr(self, name)

   ############################################################################
   ### Population Parameters                                                   
   AGENT_LOADER            = SequentialLoader
   '''Agent loader class specifying spawn sampling'''

   AGENTS = []
   '''Agent classes from which to spawn'''

   TASKS = []
   '''Tasks for which to compute rewards'''

   NMAPS                   = 1
   '''Number of maps to generate'''

   NTILE                   = 6
   #TODO: Find a way to auto-compute this
   '''Number of distinct terrain tile types'''

   NSTIM                   = 7
   '''Number of tiles an agent can see in any direction'''

   NMOB                    = None
   '''Maximum number of NPCs spawnable in the environment'''

   NENT                    = None
   '''Maximum number of agents spawnable in the environment'''

   NPOP                    = 1
   '''Number of distinct populations spawnable in the environment'''

   N_AGENT_OBS             = 100
   '''Number of distinct agent observations'''

   @property
   def TEAM_SIZE(self):
      assert not self.NENT % self.NPOP
      return self.NENT // self.NPOP

   @property
   def WINDOW(self):
      '''Size of the square tile crop visible to an agent'''
      return 2*self.NSTIM + 1

   ############################################################################
   ### Agent Parameters                                                   
   BASE_HEALTH                = 10
   '''Initial Constitution level and agent health'''

   PLAYER_SPAWN_ATTEMPTS      = None
   '''Number of player spawn attempts per tick

   Note that the env will attempt to spawn agents until success
   if the current population size is zero.'''

   def SPAWN_CONTINUOUS(self):
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

   def SPAWN_CONCURRENT(self):
      left   = self.TERRAIN_BORDER
      right  = self.TERRAIN_CENTER + self.TERRAIN_BORDER
      rrange = np.arange(left+2, right, 4).tolist()

      assert not self.TERRAIN_CENTER % 4
      per_side = self.TERRAIN_CENTER // 4
      
      lows   = (left+np.zeros(per_side, dtype=np.int)).tolist()
      highs  = (right+np.zeros(per_side, dtype=np.int)).tolist()

      s1     = list(zip(rrange, lows))
      s2     = list(zip(lows, rrange))
      s3     = list(zip(rrange, highs))
      s4     = list(zip(highs, rrange))

      return s1 + s2 + s3 + s4

   @property
   def SPAWN(self):
      return self.SPAWN_CONTINUOUS

   ############################################################################
   ### Terrain Generation Parameters
   MAP_GENERATOR          = None
   '''Specifies a user map generator. Uses default generator if unspecified.'''

   FORCE_MAP_GENERATION   = False
   '''Whether to regenerate and overwrite existing maps'''

   GENERATE_MAP_PREVIEWS  = False
   '''Whether map generation should also save .png previews (slow + large file size)'''

   MAP_PREVIEW_DOWNSCALE  = 1
   '''Downscaling factor for png previews'''

   TERRAIN_CENTER             = None
   '''Size of each map (number of tiles along each side)'''

   TERRAIN_BORDER             = 16
   '''Number of lava border tiles surrounding each side of the map'''

   @property
   def TERRAIN_SIZE(self):
      return int(self.TERRAIN_CENTER + 2*self.TERRAIN_BORDER)

   TERRAIN_FLIP_SEED          = False
   '''Whether to negate the seed used for generation (useful for unique heldout maps)'''

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
   ### Path Parameters
   PATH_ROOT            = os.path.dirname(nmmo.__file__)
   '''Global repository directory'''

   PATH_CWD             = os.getcwd()
   '''Working directory'''

   PATH_RESOURCE        = os.path.join(PATH_ROOT, 'resource')
   '''Resource directory'''

   PATH_TILE            = os.path.join(PATH_RESOURCE, '{}.png')
   '''Tile path -- format me with tile name'''

   PATH_MAPS            = None
   '''Generated map directory'''

   PATH_MAP_SUFFIX      = 'map{}/map.npy'
   '''Map file name'''


############################################################################
### Game Systems (Static Mixins)
class Resource:
   '''Resource Game System'''

   @property #Reserved flag
   def Resource(self):
      return True

   RESOURCE_BASE_RESOURCE              = 10
   '''Initial level and capacity for Hunting + Fishing resource skills'''

   RESOURCE_FOREST_CAPACITY            = 1
   '''Maximum number of harvests before a forest tile decays'''

   RESOURCE_FOREST_RESPAWN             = 0.025
   '''Probability that a harvested forest tile will regenerate each tick'''

   RESOURCE_HARVEST_RESTORE_FRACTION   = 1.0
   '''Fraction of maximum capacity restored upon collecting a resource'''

   RESOURCE_HEALTH_REGEN_THRESHOLD     = 0.5
   '''Fraction of maximum resource capacity required to regen health'''

   RESOURCE_HEALTH_RESTORE_FRACTION    = 0.1
   '''Fraction of health restored per tick when above half food+water'''

class Combat:
   '''Combat Game System'''

   @property #Reserved flag
   def Combat(self):
      return True

   COMBAT_DICE_SIDES                   = 20
   '''Number of sides for combat dice

   Attacks can only hit opponents up to the attacker's level plus
   DICE_SIDES/2. Increasing this value makes attacks more accurate
   and allows lower level attackers to hit stronger opponents'''

   COMBAT_DEFENSE_WEIGHT               = 0.3 
   '''Fraction of defense that comes from the Defense skill'''   

   COMBAT_MELEE_REACH                  = 1
   '''Reach of attacks using the Melee skill'''

   COMBAT_RANGE_REACH                  = 3
   '''Reach of attacks using the Range skill'''

   COMBAT_MAGE_REACH                   = 4
   '''Reach of attacks using the Mage skill'''

   COMBAT_FREEZE_TIME                  = 3
   '''Number of ticks successful Mage attacks freeze a target'''


class Progression:
   '''Progression Game System'''

   @property #Reserved flag
   def Progression(self):
      return True

   PROGRESSION_BASE_RESOURCE           = 10
   '''Initial level and capacity for Hunting + Fishing resource skills'''

   PROGRESSION_BASE_XP_SCALE           = 10
   '''Skill level progression speed as a multiplier of typical MMOs'''

   PROGRESSION_CONSTITUTION_XP_SCALE   = 2
   '''Multiplier on top of XP_SCALE for the Constitution skill'''

   PROGRESSION_COMBAT_XP_SCALE         = 4
   '''Multiplier on top of XP_SCALE for Combat skills'''


class NPC(Combat):
   '''NPC & Equipment Game System'''

   @property #Reserved flag
   def NPC(self):
      return True

   NPC_SPAWN_ATTEMPTS                  = 25
   '''Number of NPC spawn attempts per tick'''

   NPC_SPAWN_AGGRESSIVE                = 0.80
   '''Percentage distance threshold from spawn for aggressive NPCs'''

   NPC_SPAWN_NEUTRAL                   = 0.50
   '''Percentage distance threshold from spawn for neutral NPCs'''

   NPC_SPAWN_PASSIVE                   = 0.00
   '''Percentage distance threshold from spawn for passive NPCs'''
   
   NPC_LEVEL_MIN                       = 1
   '''Minimum NPC level'''

   NPC_LEVEL_MAX                       = None
   '''Maximum NPC level'''

   NPC_LEVEL_SPREAD                    = None
   '''Level range for NPC spawns'''


class AllGameSystems(Resource, Progression, NPC): pass

############################################################################
### Config presets
class Small(Config):
   '''A small config for debugging and experiments with an expensive outer loop'''

   PATH_MAPS               = 'maps/small' 
   MAP_PREVIEW_DOWNSCALE   = 4

   TERRAIN_LOG_INTERPOLATE_MIN = 0

   TERRAIN_CENTER          = 32
   NENT                    = 64
   NMOB                    = 32

   PLAYER_SPAWN_ATTEMPTS   = 1

   NPC_LEVEL_MAX           = 10
   NPC_LEVEL_SPREAD        = 1

class Medium(Config):
   '''A medium config suitable for most academic-scale research'''

   PATH_MAPS               = 'maps/medium' 
   MAP_PREVIEW_DOWNSCALE   = 16

   TERRAIN_CENTER          = 128
   NENT                    = 256
   NMOB                    = 128

   PLAYER_SPAWN_ATTEMPTS   = 2

   NPC_LEVEL_MAX           = 30
   NPC_LEVEL_SPREAD        = 5

class Large(Config):
   '''A large config suitable for large-scale research or fast models'''

   PATH_MAPS               = 'maps/large' 
   MAP_PREVIEW_DOWNSCALE   = 64

   TERRAIN_CENTER          = 1024
   NENT                    = 2048
   NMOB                    = 1024

   PLAYER_SPAWN_ATTEMPTS   = 16

   NPC_LEVEL_MAX           = 99
   NPC_LEVEL_SPREAD        = 10

class Default(Medium, AllGameSystems): pass
