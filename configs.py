#You removed advantage estimateion
#Advantage subtract mean works, but divide by std crashes single atn choices

#ensure there is an all 0 pad action

from pdb import set_trace as T
from forge.blade.core.config import Config
from forge.blade.lib import utils
import numpy as np

class Experiment(Config):
   MODELDIR='resource/logs/'
   EMBED   = 128
   HIDDEN  = 128
   NHEAD   = 8
   NGOD = 2
   NSWORD = 2
   NATN = 1
   KEYLEN = 4

   #EPOCHUPDATES: Number of experience steps per 
   #synchronized gradient step at the cluster level
   EPOCHUPDATES = 2**14 #Training
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

   ENTROPY = 0.01
