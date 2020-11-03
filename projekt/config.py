from forge.blade import core

class Config(core.Config):
   #Model to load. None will train from scratch
   #Baselines: recurrent, attentional, convolutional 
   #"current" will resume training custom models
   MODEL    = 'current'
   SCRIPTED = False
   RENDER   = False #Don't edit this manually; TODO: remove it

   #Model dimensions
   EMBED   = 64
   HIDDEN  = 64

   #TODO: Spawning retry logic in Player/NPC for overlapping spawns

   #Environment parameters
   NENT    = 256      #Maximum population size
   NPOP    = 1        #Number of populations
   NMOB    = 0        #Number of NPCS

   TERRAIN_SIZE    = 80 #Side dimension of each map
   TERRAIN_OCTAVES = 1  #Comment for fancy maps
   NMAPS        = 256 #Number maps to generate

   #Evaluation parameters
   EVALUATION_HORIZON = 2048


   #Agent vision range
   STIM    = 4

   #Maximum number of observed agents
   N_AGENT_OBS = 100

   #Whether to share weights across policies
   #The 1.4 baselines use one policy
   POPULATIONS_SHARE_POLICIES = True
   NPOLICIES = 1 if POPULATIONS_SHARE_POLICIES else NPOP

   #Evaluation
   LOG_DIR    = 'experiment/'
   LOG_FILE   = 'evaluation.npy'
   LOG_FIGURE = 'evaluation.html'

   #Visualization
   THEME_DIR               = 'forge/blade/systems/visualizer/'
   THEME_NAME              = 'web' #publication or web
   THEME_FILE              = 'theme_temp.json'
   THEME_WEB_INDEX         = 'index_web.html'
   THEME_PUBLICATION_INDEX = 'index_publication.html'
   PORT                    = 5006
   PLOT_WIDTH              = 1920
   PLOT_HEIGHT             = 270 
   PLOT_COLUMNS            = 4
   PLOT_TOOLS              = False
   PLOT_INTERACTIVE        = False
