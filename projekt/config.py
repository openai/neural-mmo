from pdb import set_trace as T
from forge.blade import core
import os

class Config(core.Config):
   # Model to load. None will train from scratch
   # Baselines: recurrent, attentional, convolutional
   # "current" will resume training custom models

   v                       = False

   ENV_NAME                = 'Neural_MMO'
   ENV_VERSION             = '1.5'
   NUM_WORKERS             = 4
   NUM_GPUS_PER_WORKER     = 0
   NUM_GPUS                = 1
   TRAIN_BATCH_SIZE        = 4000
   #TRAIN_BATCH_SIZE        = 400
   ROLLOUT_FRAGMENT_LENGTH = 100
   SGD_MINIBATCH_SIZE      = 128
   NUM_SGD_ITER            = 1

   MODEL        = 'current'
   SCRIPTED_BFS = False
   SCRIPTED_DP  = False
   EVALUATE     = False
   LOCAL_MODE   = False

   # Model dimensions
   EMBED  = 64
   HIDDEN = 64

   # Environment parameters
   NPOP = 1    # Number of populations #SET SHARE POLICY TRUE
   NENT = 1024 # Maximum population size
   NMOB = 1024 # Number of NPCS

   NMAPS = 256 # Number maps to generate

   #Horizons for training and evaluation
   #TRAIN_HORIZON      = 500 #This in in agent trajs
   TRAIN_HORIZON      = 1000 #This in in agent trajs
   EVALUATION_HORIZON = 2048 #This is in timesteps

   #Agent vision range
   STIM    = 7

   #Maximum number of observed agents
   N_AGENT_OBS = 100

   # Whether to share weights across policies
   # The 1.4 baselines use one policy
   POPULATIONS_SHARE_POLICIES = False
   NPOLICIES = 1 if POPULATIONS_SHARE_POLICIES else NPOP

   #Overlays
   OVERLAY_GLOBALS = False

   #Evaluation
   LOG_DIR = 'experiment/'
   LOG_FILE = 'evaluation.npy'
   LOG_FIGURE = 'evaluation.html'

   #Visualization
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
