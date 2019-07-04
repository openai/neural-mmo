from pdb import set_trace as T
import os 

from forge.blade.core import config

class Config(config.Config):
   MODELDIR = 'resource/exps' #Where to store models
   DEBUG    = True

   LOAD = False #Load model from file?
   BEST = False #If loading, most recent or highest lifetime?
   TEST = False #Update the model during run?

   NENT = 128
   NPOP = 1

   NATN    = 1    #Number of actions taken by the network
   ENTROPY = 0.01 #Entropy bonus for policy gradient loss

   HIDDEN  = 128  #Model embedding dimension
   EMBED   = 128  #Model hidden dimension
 
   NGOD   = 2  #Number of GPU optimizer servers
   NSWORD = 2  #Number of CPU rollout workers per server

   #EPOCHUPDATES: Number of experience steps per 
   #synchronized gradient step at the cluster level
   #EPOCHUPDATES = 2**14 #Training
   EPOCHUPDATES = 2**16 #Training

   #OPTIMUPDATES: Number of experience steps per 
   #optimizer server per cluster level step
   #SYNCUPDATES: Number of experience steps between 
   #syncing rollout workers to the optimizer server
   OPTIMUPDATES = EPOCHUPDATES / NGOD
   SYNCUPDATES  = 2**10

   #OPTIMBATCH: Number of experience steps per
   #.backward minibatch on optimizer servers
   #SYNCUPDATES: Number of experience steps between 
   #syncing rollout workers to the optimizer server
   OPTIMBATCH  = SYNCUPDATES * NGOD
   SYNCBATCH   = SYNCUPDATES

   #Device used on the optimizer server.
   #Rollout workers use CPU by default
   DEVICE = 'cuda:0'

   #Debug params
   if DEBUG:
      HIDDEN  = 2
      EMBED   = 2
      EPOCHUPDATES = 2**8
      SYNCUPDATES  = 2**4
      DEVICE = 'cpu:0'

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
