from pdb import set_trace as T
import numpy as np
import os

from forge.blade import core
from forge.blade.core import config
from forge.blade.systems.ai import behavior

class Achievement:
   PLAYER_KILLS_EASY       = 1
   PLAYER_KILLS_NORMAL     = 3
   PLAYER_KILLS_HARD       = 6

   EQUIPMENT_EASY          = 1
   EQUIPMENT_NORMAL        = 10
   EQUIPMENT_HARD          = 20

   EXPLORATION_EASY        = 32
   EXPLORATION_NORMAL      = 64
   EXPLORATION_HARD        = 127

   FORAGING_EASY           = 20
   FORAGING_NORMAL         = 35
   FORAGING_HARD           = 50

   REWARD_ACHIEVEMENT      = False
   ACHIEVEMENT_SCALE       = 1.0/15.0

class Base(core.Config, Achievement):
   '''Base config for RLlib Models

   Extends core Config, which contains environment, evaluation,
   and non-RLlib-specific learning parameters'''
   
   #Hardware Scale
   NUM_GPUS_PER_WORKER     = 0
   NUM_GPUS                = 1
   NUM_WORKERS             = 6
   LOCAL_MODE              = False
   LOAD                    = True

   #Memory/Batch Scale
   TRAIN_EPOCHS            = 10000
   TRAIN_BATCH_SIZE        = 256 * NUM_WORKERS #Bug? This gets doubled
   ROLLOUT_FRAGMENT_LENGTH = 256
   LSTM_BPTT_HORIZON       = 16
   SGD_MINIBATCH_SIZE      = min(256, TRAIN_BATCH_SIZE)
   NUM_SGD_ITER            = 1

   #Model
   N_AGENT_OBS             = 100
   NPOLICIES               = 1
   HIDDEN                  = 64
   EMBED                   = 64

   @property
   def MODEL(self):
      return self.__class__.__name__

   #Scripted model parameters
   SCRIPTED                = None
   SCRIPTED_BACKEND        = 'dijkstra' #Or 'dynamic_programming'
   SCRIPTED_EXPLORE        = True       #Intentional exploration

   #Reward
   COOP                    = False
   TEAM_SPIRIT             = 0.0


class LargeMaps(Base):
   '''Large scale Neural MMO training setting

   Features up to 1000 concurrent agents and 1000 concurrent NPCs,
   1km x 1km maps, and 5/10k timestep train/eval horizons

   This is the default setting as of v1.5 and allows for large
   scale multiagent research even on relatively modest hardware'''

   #Path settings
   PATH_MAPS               = core.Config.PATH_MAPS_LARGE

   #Horizon
   TRAIN_HORIZON           = 8192
   EVALUATION_HORIZON      = 8192

   #Scale
   NENT                    = 1024
   NMOB                    = 1024


class SmallMaps(Base):
   '''Small scale Neural MMO training setting

   Features up to 128 concurrent agents and 32 concurrent NPCs,
   60x60 maps (excluding the border), and 1000 timestep train/eval horizons.
   
   This setting is modeled off of v1.1-v1.4 It is appropriate as a quick train
   task for new ideas, a transfer target for agents trained on large maps,
   or as a primary research target for PCG methods.'''

   #Path settings
   PATH_MAPS               = core.Config.PATH_MAPS_SMALL

   #Horizon
   TRAIN_HORIZON           = 1024
   EVALUATION_HORIZON      = 1024

   #Scale
   TERRAIN_CENTER          = 128
   NENT                    = 256
   NMOB                    = 128

   PLAYER_SPAWN_ATTEMPTS   = 2

   #NPC parameters
   NPC_LEVEL_MAX           = 30
   NPC_LEVEL_SPREAD        = 5

class Debug(SmallMaps, config.AllGameSystems):
   '''Debug Neural MMO training setting

   A version of the SmallMap setting with greatly reduced batch parameters.
   Only intended as a tool for identifying bugs in the model or environment'''
   MODEL                   = None
   LOCAL_MODE              = True
   NUM_WORKERS             = 1

   SGD_MINIBATCH_SIZE      = 100
   TRAIN_BATCH_SIZE        = 400
   TRAIN_HORIZON           = 200
   EVALUATION_HORIZON      = 50

   HIDDEN                  = 2
   EMBED                   = 2

###NeurIPS Experiments
class SmallMultimodalSkills(SmallMaps, config.AllGameSystems):
   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

class LargeMultimodalSkills(LargeMaps, config.AllGameSystems):
   MODEL = 'SmallMultimodalSkills16384Map'

class MagnifyExploration(SmallMaps, config.Resource, config.Progression):
   pass
class Population4(MagnifyExploration):
   NENT  = 256
class Population32(MagnifyExploration):
   NENT  = 256
class Population256(MagnifyExploration):
   NENT  = 256

class DomainRandomization16384(SmallMaps, config.AllGameSystems):
   TERRAIN_TRAIN_MAPS=16384
class DomainRandomization256(SmallMaps, config.AllGameSystems):
   TERRAIN_TRAIN_MAPS=256
class DomainRandomization32(SmallMaps, config.AllGameSystems):
   TERRAIN_TRAIN_MAPS=32
class DomainRandomization1(SmallMaps, config.AllGameSystems):
   TERRAIN_TRAIN_MAPS=1

class TeamBased(MagnifyExploration, config.Combat):
   NENT                    = 128
   NPOP                    = 32
   COOP                    = True
   TEAM_SPIRIT             = 0.5

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

###Reserved for an upcoming competition
class Competition(config.AllGameSystems):
   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

class CompetitionRound1(Competition, SmallMaps):
   NENT                    = 128
   NPOP                    = 1
   COOP                    = False

class CompetitionRound2(Competition, SmallMaps):
   NENT                    = 128
   NPOP                    = 16
   COOP                    = True

class CompetitionRound3(Competition, LargeMaps):
   NENT                    = 128
   NPOP                    = 32
   COOP                    = True
