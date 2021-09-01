from pdb import set_trace as T

from neural_mmo.forge.blade import core
from neural_mmo.forge.blade.core import config
from neural_mmo.forge.blade.io.stimulus.static import Stimulus
from neural_mmo.forge.trinity.scripted import baselines
from neural_mmo.forge.trinity.agent import Agent
from neural_mmo.forge.blade.systems.ai import behavior
from projekt import rllib_wrapper

class RLlibConfig:
   '''Base config for RLlib Models

   Extends core Config, which contains environment, evaluation,
   and non-RLlib-specific learning parameters'''

   @property
   def MODEL(self):
      return self.__class__.__name__

   #Checkpointing. Resume will load the latest trial, e.g. to continue training
   #Restore (overrides resume) will force load a specific checkpoint (e.g. for rendering)
   RESUME      = False
   RESTORE     = 'experiments/CompetitionRound1/Dev_9fe1/checkpoint_001000/checkpoint-1000'

   #Policy specification
   AGENTS      = [Agent]
   EVAL_AGENTS = [baselines.Meander, baselines.Forage, baselines.Combat, Agent]
   EVALUATE    = False #Reserved param

   #Hardware and debug
   NUM_WORKERS             = 1
   NUM_GPUS_PER_WORKER     = 0
   NUM_GPUS                = 1
   EVALUATION_NUM_WORKERS  = 3
   LOCAL_MODE              = False
   LOG_LEVEL               = 1

   #Training and evaluation settings
   EVALUATION_INTERVAL     = 1
   EVALUATION_NUM_EPISODES = 3
   EVALUATION_PARALLEL     = True
   TRAINING_ITERATIONS     = 1000
   KEEP_CHECKPOINTS_NUM    = 5
   CHECKPOINT_FREQ         = 1
   LSTM_BPTT_HORIZON       = 16
   NUM_SGD_ITER            = 1

   #Model
   SCRIPTED                = None
   N_AGENT_OBS             = 100
   NPOLICIES               = 1
   HIDDEN                  = 64
   EMBED                   = 64

   #Reward
   TEAM_SPIRIT             = 0.0
   ACHIEVEMENT_SCALE       = 1.0/15.0


class LargeMaps(core.Config, RLlibConfig, config.AllGameSystems):
   '''Large scale Neural MMO training setting

   Features up to 1000 concurrent agents and 1000 concurrent NPCs,
   1km x 1km maps, and 5/10k timestep train/eval horizons

   This is the default setting as of v1.5 and allows for large
   scale multiagent research even on relatively modest hardware'''

   #Memory/Batch Scale
   NUM_WORKERS             = 14
   TRAIN_BATCH_SIZE        = 64 * 256 * NUM_WORKERS
   ROLLOUT_FRAGMENT_LENGTH = 32
   SGD_MINIBATCH_SIZE      = 128

   #Horizon
   TRAIN_HORIZON           = 8192
   EVALUATION_HORIZON      = 8192


class SmallMaps(RLlibConfig, config.AllGameSystems, config.SmallMaps):
   '''Small scale Neural MMO training setting

   Features up to 128 concurrent agents and 32 concurrent NPCs,
   60x60 maps (excluding the border), and 1000 timestep train/eval horizons.
   
   This setting is modeled off of v1.1-v1.4 It is appropriate as a quick train
   task for new ideas, a transfer target for agents trained on large maps,
   or as a primary research target for PCG methods.'''

   #Memory/Batch Scale
   NUM_WORKERS             = 28
   TRAIN_BATCH_SIZE        = 64 * 256 * NUM_WORKERS
   ROLLOUT_FRAGMENT_LENGTH = 256
   SGD_MINIBATCH_SIZE      = 128
 
   #Horizon
   TRAIN_HORIZON           = 1024
   EVALUATION_HORIZON      = 1024


class Debug(SmallMaps, config.AllGameSystems):
   '''Debug Neural MMO training setting

   A version of the SmallMap setting with greatly reduced batch parameters.
   Only intended as a tool for identifying bugs in the model or environment'''
   LOAD                    = False
   LOCAL_MODE              = True
   NUM_WORKERS             = 1

   SGD_MINIBATCH_SIZE      = 100
   TRAIN_BATCH_SIZE        = 400
   TRAIN_HORIZON           = 200
   EVALUATION_HORIZON      = 50

   HIDDEN                  = 2
   EMBED                   = 2


### AICrowd competition settings
class CompetitionRound1(config.Achievement, SmallMaps):
   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

   NENT                    = 128
   NPOP                    = 1

class CompetitionRound2(config.Achievement, SmallMaps):
   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

   NENT                    = 128
   NPOP                    = 16
   COOPERATIVE             = True

class CompetitionRound3(config.Achievement, LargeMaps):
   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

   NENT                    = 1024
   NPOP                    = 32
   COOPERATIVE             = True


### NeurIPS Experiments
class SmallMultimodalSkills(SmallMaps, config.AllGameSystems): pass
class LargeMultimodalSkills(LargeMaps, config.AllGameSystems): pass


class MagnifyExploration(SmallMaps, config.Resource, config.Progression):
   pass
class Population4(MagnifyExploration):
   NENT  = 4
class Population32(MagnifyExploration):
   NENT  = 32
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
   COOPERATIVE             = True
   TEAM_SPIRIT             = 0.5

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT
