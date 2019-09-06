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
   DEBUG    = False #Whether to run with debug settings

   LOAD = False #Load model from file?
   BEST = False #If loading, most recent or highest lifetime?
   TEST = False #Update the model during run?

   #Typically overriden in Forge.py
   NENT = 1  #Maximum population size
   NPOP = 1  #Number of populations

   NATN    = 1    #Number of actions taken by the network (deprecated)
   ENTROPY = 0.01 #Entropy bonus for policy gradient loss

   HIDDEN  = 128  #Model embedding dimension
   EMBED   = 16   #Model hidden dimension
 
   #NGOD   = 6 #Number of environment servers
   NGOD   = 6 #Number of environment servers
   NSWORD = 1 #Number of clients per server

   #Number of experience steps before
   #syncronizing at each hardware layer
   CLUSTER_UPDATES = 4096
   SERVER_UPDATES  = CLUSTER_UPDATES / NGOD
   CLIENT_UPDATES  = 256
   CLIENT_TICKS = 64

   #Hardware specification
   #DEVICE = 'cpu:0'

   #Gradient based optimization parameters
   LR         = 1e-3
   DECAY      = 1e-5
   VAL_WEIGHT = 0.25

   #Experimental population based training parameters
   #Disabled and not currently functional -- avoid modification
   POPOPT   = False
   PERMPOPS = 4
   PERMVAL  = 1e-2

   #Debug params
   if DEBUG:
      HIDDEN  = 2
      EMBED   = 2

      CLUSTER_UPDATES = 32
      SERVER_UPDATES  = CLUSTER_UPDATES / NGOD
      CLIENT_UPDATES  = 16

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
