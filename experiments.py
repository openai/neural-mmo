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

   #Top experiment is 1 pop, bottom is 8
   #Top is on vision 35
   ###############################Saving and logging locations
   MODELDIR  = 'resource/exps'  #Where to store models
   HOST      = 'localhost'      #Host for client
   STAT_FILE = 'stats.txt'      #Run statistics log file
   DEVICE    = 'cpu'            #Hardware specification

   ###############################Train/test mode settings
   DEBUG     = False            #Whether to run with debug settings
   LOAD      = False            #Load model from file?
   BEST      = False            #If loading, most recent or highest lifetime?
   TEST      = False            #Update the model during run?

   ###############################Distributed infrastructure config
   NGOD    = 1                  #Number of environment servers
   NSWORD  = 1                  #Number of clients per server
   NCORE   = NGOD*(NSWORD + 1)  #Total number of cores

   #_ = 256
   _ = 4096
   CLUSTER_UPDATES = _          #Number of samples per optim
   SERVER_UPDATES  = _ // NGOD  #step at each hardware layer

   ###############################Population and network sizes
   NENT    = 128                #Maximum population size
   NPOP    = 1                  #Number of populations

   HIDDEN  = 32                 #Model embedding dimension
   EMBED   = 32                 #Model hidden dimension
 
   ###############################Gradient based optimization parameters
   LR         = 3e-4            #Learning rate
   DECAY      = 1e-5            #Weight decay
   GRAD_CLIP  = 5.0             #Gradient absolute value clip threshold
   DISCOUNT   = 0.95            #Reward discount factor
   VAL_WEIGHT = 0.5             #Value function loss weighting
   ENTROPY    = 0.000           #Entropy bonus strength
   #ENTROPY    = 0.001           #Entropy bonus strength

   ###############################Per agent logging settings
   SAVE_BLOBS = False           #Log at all? (IO/comms intensive)
   BLOB_FRAC  = 0.1             #What fraction of blobs to log?

   ###############################Experimental population based training
   ###############################parameters -- not currently used
   POPOPT   = False             #Whether to enable
   PERMPOPS = 4                 #Number of permutations
   PERMVAL  = 1e-2              #Permutation strength

   #Parameter overrides for debugging
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
