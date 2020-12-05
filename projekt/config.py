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
   EVALUATE     = False
   LOCAL_MODE   = False

   # Model dimensions
   EMBED  = 64
   HIDDEN = 64

   # Environment parameters
   NPOP = 1    # Number of populations
   NENT = 256  # Maximum population size
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

#Small map preset
class SmallMap(Config):
   MODEL                   = 'small-map'

   NENT                    = 128
   NMOB                    = 32

   TERRAIN_MODE            = 'contract'
   TERRAIN_LERP            = False

   TERRAIN_SIZE            = 80 
   TERRAIN_OCTAVES         = 1
   TERRAIN_FOREST_LOW      = 0.30
   TERRAIN_FOREST_HIGH     = 0.75
   TERRAIN_GRASS           = 0.715
   TERRAIN_ALPHA           = -0.025
   TERRAIN_BETA            = 0.035

   TERRAIN_DIR             = Config.TERRAIN_DIR_SMALL
   ROOT                    = os.path.join(os.getcwd(), TERRAIN_DIR, 'map')

   INVERT_WILDERNESS       = True
   WILDERNESS              = False

   NPC_LEVEL_MAX           = 35
   NPC_LEVEL_SPREAD        = 5

   NPC_SPAWN_PASSIVE       = 0.00
   NPC_SPAWN_NEUTRAL       = 0.60
   NPC_SPAWN_AGGRESSIVE    = 0.80
