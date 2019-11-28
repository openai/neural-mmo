from pdb import set_trace as T
import os 

from forge.blade.core import config

class Config(config.Config):
   '''Configuration specification for Neural MMO experiements

   Additional parameters can be found in the forge/blade configuration.
   These represent game parameters -- you can edit them, but beware that
   this can significantly alter the environment.

   All parameters can also be overridden at run time.
   See Forge.py for an example'''

   MODELDIR = 'resource/exps' #Where to store models
   HOST     = 'localhost'     #Host for client
   DEBUG    = True            #Whether to run with debug settings

   LOAD = False #Load model from file?
   BEST = False #If loading, most recent or highest lifetime?
   TEST = False #Update the model during run?

   #Typically overriden in Forge.py
   NENT = 128 #Maximum population size
   NPOP = 8  #Number of populations

   HIDDEN  = 32 #Model embedding dimension
   EMBED   = 32 #Model hidden dimension
 
   #It appears we have exploding gradients?
   NGOD   = 6                   #Number of environment servers
   NSWORD = 1                   #Number of clients per server
   NCORE  = NGOD * (NSWORD + 1) #Total number of cores

   #Number of experience steps before
   #syncronizing at each hardware layer
   CLUSTER_UPDATES = 4096
   SERVER_UPDATES  = CLUSTER_UPDATES // NGOD

   #Hardware specification
   #DEVICE = 'cpu:0'

   #Gradient based optimization parameters
   LR         = 3e-4
   DECAY      = 1e-5
   GRAD_CLIP  = 5.0
   VAL_WEIGHT = 0.5
   ENTROPY    = 0.0

   #Per agent logging
   SAVE_BLOBS = False #Log at all? (IO/comms intensive)
   BLOB_FRAC  = 0.1   #What fraction of blobs to log?

   #Experimental population based training parameters
   #Disabled and not currently functional -- avoid modification
   POPOPT   = False
   PERMPOPS = 4
   PERMVAL  = 1e-2

   #Stats
   STAT_FILE = 'stats.txt'

   #Debug params
   if DEBUG:
      NGOD = 1
      NSWORD = 1

      HIDDEN  = 4
      EMBED   = 4

      CLUSTER_UPDATES = 128
      SERVER_UPDATES  = CLUSTER_UPDATES

class Experiment:
   '''Manages file structure for experiments'''
   def mkdirs(self, path):
      if os.path.exists(path):
         return
      os.makedirs(path)
      
   def __init__(self, name, conf):
      ROOT = os.path.join(
         os.path.dirname(__file__), 
         conf.MODELDIR, name, '')

      for path in 'model train test'.split():
         self.mkdirs(os.path.join(ROOT, path))

      #Extend model directory
      self.MODELDIR = os.path.join(ROOT, 'model')
      self.config = conf
      self.name = name

      #Remove old stats file
      path = os.path.join(self.MODELDIR, conf.STAT_FILE)
      with open(path, 'w') as f: pass

   def init(self, **kwargs):
      assert 'MODELDIR' not in kwargs 
      conf = self.config(MODELDIR=self.MODELDIR, **kwargs)
      
      print('Experiment: ', self.name, 
         '-->   NENT: ', conf.NENT, 
         ', NPOP: ', conf.NPOP)

      return conf


   def makeExps():
      #Training runs
      for label, conf in zip(names, confs):
         for sz in szs:
            name = prefix + label + str(sz)
            makeExp(name, conf, sz)
      return exps
