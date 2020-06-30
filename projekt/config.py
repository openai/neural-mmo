from forge.blade import core

class Config(core.Config):
   #Computing the global value map takes ~20 seconds
   #Disabling this will speed up server start
   COMPUTE_GLOBAL_VALUES = True

   #Model to load. None will train from scratch
   #Baselines: recurrent, attentional, convolutional 
   #"current" will resume training custom models
   MODEL   = 'recurrent'
   RENDER  = True

   NENT    = 256 #Maximum population size
   NPOP    = 1   #Number of populations
   NMAPS   = 256 #Number of game maps

   #Model dimensions
   EMBED   = 64
   HIDDEN  = 64

   #Agent vision range
   STIM    = 4
   WINDOW  = 9  #Reduced from 15

   #Maximum number of observed agents
   N_AGENT_OBS = 15

   #Whether to share weights across policies
   #The 1.4 baselines use one policy
   POPULATIONS_SHARE_POLICIES = True
   NPOLICIES = 1 if POPULATIONS_SHARE_POLICIES else NPOP
