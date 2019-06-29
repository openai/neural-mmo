from pdb import set_trace as T
import os 

from forge.blade.core.config import Config

class Experiment(Config):
   MODELDIR ='resource/logs/'

   NGOD   = 2  #Number of GPU optimizer servers
   NSWORD = 2  #Number of CPU rollout workers per server
 
   HIDDEN = EMBED = 128 #Model embedding and hidden dimensions
   ENTROPY = 0.01       #Entropy bonus for policy gradient loss

   NATN   = 1 #Number of actions taken by the network

   #EPOCHUPDATES: Number of experience steps per 
   #synchronized gradient step at the cluster level
   #EPOCHUPDATES = 2**14 #Training
   EPOCHUPDATES = 2**8  #Local debug

   #OPTIMUPDATES: Number of experience steps per 
   #optimizer server per cluster level step
   #SYNCUPDATES: Number of experience steps between 
   #syncing rollout workers to the optimizer server
   OPTIMUPDATES = EPOCHUPDATES / NGOD
   SYNCUPDATES  = OPTIMUPDATES / 2**4

   #OPTIMBATCH: Number of experience steps per
   #.backward minibatch on optimizer servers
   #SYNCUPDATES: Number of experience steps between 
   #syncing rollout workers to the optimizer server
   OPTIMBATCH  = SYNCUPDATES * NGOD
   SYNCBATCH   = SYNCUPDATES

   #Device used on the optimizer server.
   #Rollout workers use CPU by default
   DEVICE = 'cuda:0'

#A dumb but simple way to have user-specific experiment
#configuration when working off of the same fork
USER = 'your-username'
if USER == 'your-username':
   #Thousandth

   load = False #Load model from file?
   best = False #If loading, most recent or highest lifetime?
   test = False #Update the model during run?

   #Name prefix for experiments
   prefix, exps = 'demo-', {}

   #You can generate multiple experiment
   #configurations at once. The demo
   #hardcodes the name for the size 128
   #experiment in Forge.py
   szs = (16, 32, 64, 128)
   names = ['baseline-']
   confs = [Experiment]

   def makeExp(name, conf, sz):
      NENT, NPOP = sz, 1
      #NENT, NPOP = sz, sz//16
      ROOT = os.path.dirname(__file__)
      ROOT += '/resource/exps/' + name + '/'
      try:
         os.mkdir(ROOT)
         os.mkdir(ROOT + 'model')
         os.mkdir(ROOT + 'train')
         os.mkdir(ROOT + 'test')
      except FileExistsError:
         pass
      MODELDIR = ROOT + 'model/'

      exps[name] = conf( 
            NENT=NENT, NPOP=NPOP,
            MODELDIR=MODELDIR,
            LOAD=load, BEST=best, TEST=test)
      print(name, ', NENT: ', NENT, ', NPOP: ', NPOP)

   def makeExps():
      #Training runs
      for label, conf in zip(names, confs):
         for sz in szs:
            name = prefix + label + str(sz)
            makeExp(name, conf, sz)
          
   #Make configs
   makeExps()
