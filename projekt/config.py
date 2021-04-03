from pdb import set_trace as T
import numpy as np
import os

from forge.blade import core
from forge.blade.systems.ai import behavior

class Base(core.Config):
   '''Base config for RLlib Models

   Extends core Config, which contains environment, evaluation,
   and non-RLlib-specific learning parameters'''
   
   #Hardware Scale
   NUM_WORKERS             = 6
   NUM_GPUS_PER_WORKER     = 0
   NUM_GPUS                = 1
   LOCAL_MODE              = False

   TRAIN_EPOCHS            = 500
   LOAD                    = True

   #Memory/Batch Scale
   TRAIN_BATCH_SIZE        = 256 * NUM_WORKERS #Bug? This gets doubled
   ROLLOUT_FRAGMENT_LENGTH = 256
   LSTM_BPTT_HORIZON       = 16

   #Optimization Scale
   SGD_MINIBATCH_SIZE      = min(512, TRAIN_BATCH_SIZE)
   NUM_SGD_ITER            = 1

   #Model Parameters 
   #large-map:        Large maps baseline
   #small-map:        Small maps baseline
   #scripted-combat:  Scripted with combat
   #scripted-forage:  Scripted without combat
   #current:          Resume latest checkpoint
   #None:             Train from scratch
   MODEL                   = 'current'
   N_AGENT_OBS             = 100
   NPOLICIES               = 1
   HIDDEN                  = 64
   EMBED                   = 64

   #Scripted model parameters
   SCRIPTED_BACKEND        = 'dijkstra' #Or 'dynamic_programming'
   SCRIPTED_EXPLORE        = True       #Intentional exploration

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


class LargeMaps(Base):
   '''Large scale Neural MMO training setting

   Features up to 1000 concurrent agents and 1000 concurrent NPCs,
   1km x 1km maps, and 5/10k timestep train/eval horizons

   This is the default setting as of v1.5 and allows for large
   scale multiagent research even on relatively modest hardware'''

   NAME                    = __qualname__
   MODEL                   = 'large-map'

   PATH_MAPS               = core.Config.PATH_MAPS_LARGE

   TRAIN_HORIZON           = 5000
   EVALUATION_HORIZON      = 10000

   NENT                    = 1024
   NMOB                    = 1024


class SmallMaps(Base):
   '''Small scale Neural MMO training setting

   Features up to 128 concurrent agents and 32 concurrent NPCs,
   60x60 maps (excluding the border), and 1000 timestep train/eval horizons.
   
   This setting is modeled off of v1.1-v1.4 It is appropriate as a quick train
   task for new ideas, a transfer target for agents trained on large maps,
   or as a primary research target for PCG methods.'''

   NAME                    = __qualname__
   MODEL                   = 'small-map'
   SCRIPTED_EXPLORE        = False

   #Horizon
   TRAIN_HORIZON           = 1024
   EVALUATION_HORIZON      = 1024

   #Scale
   TERRAIN_CENTER          = 128
   NENT                    = 128
   NMOB                    = 128

   #Path settings
   PATH_MAPS               = core.Config.PATH_MAPS_SMALL
   PATH_ROOT               = os.path.join(os.getcwd(), PATH_MAPS, 'map')

   #Entity spawning parameters
   PLAYER_SPAWN_ATTEMPTS   = 1
   NPC_LEVEL_MAX           = 30
   NPC_LEVEL_SPREAD        = 5
   NPC_SPAWN_PASSIVE       = 0.00
   NPC_SPAWN_NEUTRAL       = 0.60
   NPC_SPAWN_AGGRESSIVE    = 0.80

class BattleRoyale(SmallMaps):
   NPOP                    = 16
   NPOLICIES               = 1
   #N_TRAIN_MAPS            = 1
   #N_EVAL_MAPS             = 0
    

   def SPAWN_BR(self):
      left   = self.TERRAIN_BORDER
      right  = self.TERRAIN_CENTER + self.TERRAIN_BORDER
      rrange = np.arange(left+2, right, 4).tolist() 

      lows   = (left+np.zeros(32, dtype=np.int)).tolist()
      highs  = (right+np.zeros(32, dtype=np.int)).tolist()

      s1     = list(zip(rrange, lows))
      s2     = list(zip(lows, rrange))
      s3     = list(zip(rrange, highs))
      s4     = list(zip(highs, rrange))

      return s1 + s2 + s3 + s4

class Debug(SmallMaps):
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


