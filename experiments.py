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

   ###############################Saving and logging locations
   MODELDIR  = 'resource/exps'  #Where to store models
   HOST      = 'localhost'      #Host for client
   STAT_FILE = 'run_stats.txt'  #Run statistics log file
   DEVICE    = 'cpu'            #Hardware specification

   ###############################Train/test mode settings
   DEBUG     = False            #Whether to run with debug settings
   LOAD      = True             #Load model from file?
   BEST      = True             #If loading, most recent or highest lifetime?
   TEST      = True             #Update the model during run?

   ###############################Distributed infrastructure config
   NGOD    = 12                 #Number of environment servers
   NSWORD  = 1                  #Number of clients per server
   NCORE   = NGOD*(NSWORD + 1)  #Total number of cores

   _ = 16384
   CLUSTER_UPDATES = _          #Number of samples per optim
   SERVER_UPDATES  = _ // NGOD  #step at each hardware layer

   ###############################Population and network sizes
   NENT    = 128                #Maximum population size
   NPOP    = 8                  #Number of populations

   HIDDEN  = 32                 #Model embedding dimension
   EMBED   = 32                 #Model hidden dimension
 
   ###############################Gradient based optimization parameters
   LR         = 3e-4            #Learning rate
   DECAY      = 1e-5            #Weight decay
   GRAD_CLIP  = 5.0             #Gradient absolute value clip threshold
   GAMMA      = 0.95            #Reward discount factor
   LAMBDA     = 0.96            #GAE discount factor
   HORIZON    = 8               #GAE horizon
   PG_WEIGHT  = 1.0             #Policy gradient loss weighting
   VAL_WEIGHT = 0.5             #Value function loss weighting
   ENTROPY    = 0.025           #Entropy bonus strength

   ###############################Per agent logging settings
   SAVE_BLOBS = False           #Log at all? (IO/comms intensive)
   BLOB_FRAC  = 0.1             #What fraction of blobs to log?

   ###############################Experimental population based training
   ###############################parameters -- not currently used
   POPOPT   = False             #Whether to enable
   PERMPOPS = 4                 #Number of permutations
   PERMVAL  = 1e-2              #Permutation strength

   ############################################LOGGING parameters
   LOG         = False                       #Whether to enable data logging
   LOAD_EXP    = False                       #Whether to load from file
   NAME        = 'log'                       #Name of file to load/log to
   HISTORY_LEN = 0                           #Length of graph history
   TITLE       = 'Neural MMO Training Curve' #Graph title
   XAXIS       = 'Training Epoch'            #Label of xaxis data values
   YLABEL      = 'Agent Lifetime'            #Label of data values
   TITLE       = 'Neural MMO Data'            #Title of graph
   SCALES      = [1, 10, 100, 1000]          #Plot time scale

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
