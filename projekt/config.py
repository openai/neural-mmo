from forge.blade import core

class Config(core.Config):
   #Program level args
   COMPUTE_GLOBAL_VALUES = False
   RENDER                = False

   NENT    = 256
   NPOP    = 1

   NREALM  = 256
   NWORKER = 12
   NMAPS   = 256

   #Set this high enough that you can always attack
   #Probably should sort by distance
   ENT_OBS = 15

   EMBED   = 64
   HIDDEN  = 64
   STIM    = 4
   WINDOW  = 9
   #WINDOW  = 15
