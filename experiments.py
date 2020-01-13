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
   STAT_FILE = 'stats31.txt'    #Run statistics log file
   DEVICE    = 'cpu'            #Hardware specification

   #All run defaults: 12 gods, 16384 batch
   #0: 5e-3 ent (sanity) 18.72
   #1: 1e-2 ent: 16.66
   #2: 2.5e-2: 17.91
   #3: 5e-2: 17.58
   #4: 1e-1: 16.85
   #7: 12 god sanity with double batch and 1e-3 ent
   #8: 5e-3 ent
   #9: 2e-4 ent
   #10: 1e-3 ent, fc-relu-fc hidden
   #All run defaults: 12 gods, 16384 batch, 2.5e-2 entropy
   #11: 1 pop: 17.95
   #12: 1 pop, old dotrelublock, remove hidden net, 18.20
   #13: 1 pop, remove hidden net, 18.09
   #14: orig hypers (batch doubled)
   #16: orig hypers, same fucking network, orig stupid retain_gradients and stale data because why the hell not? 15.68
   #17: Above but fixed the input net (conv and fc were flipped, so was conving over ents). Note that there was no normalization in orig 15.01
   #18: Orig net with new hypers. Consistency isnt going to cut it: 18.22
   #19: 8 pops: 18.23
   #20: discount 0.98: 17.77
   #21: discount 0.90: 17.56
   #22: discount 0.95: 8.108, avg optim vals
   #23: Non mean centered val func. discount 0.95, avg optim vals: 8.293
   #24: Non mean centered val func, no avg optim vals: 18.04
   #25: Fix reward partial trajs, discount .95, no avg: 18.15
   #26: GAE: 8.084
   #29: Discount; sanity
   #30: Stale data: 23.33 :) works :)
   #31: No entropy
   ###############################Train/test mode settings
   DEBUG     = False            #Whether to run with debug settings
   LOAD      = False            #Load model from file?
   BEST      = False            #If loading, most recent or highest lifetime?
   TEST      = False            #Update the model during run?

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
