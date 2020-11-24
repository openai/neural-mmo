from pdb import set_trace as T
from forge.blade import core
import os

class Config(core.Config):
   # Model to load. None will train from scratch
   # Baselines: recurrent, attentional, convolutional
   # "current" will resume training custom models
   MODEL        = 'current'
   SCRIPTED_BFS = False
   SCRIPTED_DP  = False
   RENDER       = False  # Don't edit this manually; TODO: remove it

   # Model dimensions
   EMBED  = 64
   HIDDEN = 64

   # Environment parameters
   NENT = 256  # Maximum population size
   NPOP = 1    # Number of populations
   NMOB = 32   # Number of NPCS

   NMAPS = 256 # Number maps to generate

   # Evaluation parameters
   EVALUATION_HORIZON = 2048

   #Agent vision range
   STIM    = 4

   #Maximum number of observed agents
   N_AGENT_OBS = 100

   # Whether to share weights across policies
   # The 1.4 baselines use one policy
   POPULATIONS_SHARE_POLICIES = True
   NPOLICIES = 1 if POPULATIONS_SHARE_POLICIES else NPOP

   # Evaluation
   LOG_DIR = 'experiment/'
   LOG_FILE = 'evaluation.npy'
   LOG_FIGURE = 'evaluation.html'

   # Visualization
   THEME_DIR = 'forge/blade/systems/visualizer/'
   THEME_NAME = 'web'  # publication or web
   THEME_FILE = 'theme_temp.json'
   THEME_WEB_INDEX = 'index_web.html'
   THEME_PUBLICATION_INDEX = 'index_publication.html'
   PORT = 5006
   PLOT_WIDTH = 1920
   PLOT_HEIGHT = 270
   PLOT_COLUMNS = 4
   PLOT_TOOLS = False
   PLOT_INTERACTIVE = False

#Map Size Presets
class SmallMap(Config):
   TERRAIN_SIZE       = 80 
   TERRAIN_OCTAVES    = 1
   TERRAIN_FOREST_LOW = 0.30

   MODEL              = 'small-map'
   TERRAIN_DIR        = Config.TERRAIN_DIR_SMALL
   ROOT               = os.path.join(os.getcwd(), TERRAIN_DIR, 'map')

   NPC_LEVEL_MAX      = 40
   NPC_LEVEL_SPREAD   = 10

class LargeMap(Config):
   MODEL              = 'large-map'
   TERRAIN_DIR        = Config.TERRAIN_DIR_LARGE
   ROOT               = os.path.join(os.getcwd(), TERRAIN_DIR, 'map')

#Battle Royale Presets
class BattleRoyale(Config):
   pass

class SmallBR(SmallMap, BattleRoyale):
   pass

class LargeBR(LargeMap, BattleRoyale):
   pass

#MMO Presets
class MMO(Config):
   TERRAIN_INVERT    = True

class SmallMMO(SmallMap, MMO):
   pass

class LargeMMO(LargeMap, MMO):
   pass


