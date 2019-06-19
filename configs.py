# forge/blade/io/ - action, stimulus x static, dynamic
# forge/ethyr/torch/netgen - stim, action
# forge/ethyr/torch/modules/transformer 
# forge/ethyr/torch/ - loss, optim 
# forge/trinity - god, sword, ann

# Potential bugs:
# Bad architecture
# Bad indexing in action selection
# Bad rollout assembly
# 1 population
# 
#Value of always North policy is 20-25
#Build a mini pipeline that trains, keep expanding until something breaks it

#You removed advantage estimateion
#Check key in action/dynamic serialize
#ConstDiscrete gives 25 lifetime in 25 steps
#VariableDiscrete gives 15 tops. Figure out why
#Stillusing only three tiles / 1 entity

#streamline static actions
#streamline dynamic action next

#ensure there is an all 0 pad action

#You found out:
#Variable works without hard coding the keys
#Training deeper (2 layers) works but takes >200+ epochs to get 20+ consistent
#Attk does not crash but makes training go to 10
#Advantage subtract mean works, but divide by std crashes single atn choices

#Transformer does not work at all. Linear does.
#You also disabled multiple ents. Issue for attack
#You enabled variable discrete
#Attack might actually work -- variable issue could be the transformer

#Training epochs has gone up to ~50
#Linear env transformer out works
#The softmax causes issues. Removed for now
#Env stim only works with linear
#With linear stim:
#Multiply x*kv is a good baseline (40 lifetime in 30 epochs). Still does not work with env
#Adding an fc layer doesnt really help
#kv alone (no key) gets sub 30ish in 50 epochs

#Current config worked over 2-3 days with 53 best, 50 avg

#Replaced mean with max. This is a small test vs dota model.
#The next step will be to add type embeddings (as opposed to value embeddings)
#To all input stat and action types
#This is to get around concat, which may or may not be possible

from pdb import set_trace as T
from forge.blade.core.config import Config
from forge.blade.lib import utils
import numpy as np

class Experiment(Config):
   MODELDIR='resource/logs/'
   EMBED   = 128
   HIDDEN  = 128
   NHEAD   = 8
   NGOD = 1
   NATN = 1
   #NSWORD = 2
   KEYLEN = 4

   '''
   NROLLOUTS = NGOD * 400#10 #Rollouts per gradient step
   SYNCUPDATES = 1024#100 #Number of data to sync
   DEVICE = 'cuda:0'

   #CPU Debug mode
   #DEVICE = 'cpu:0'
 
   '''
   #CPU Dev mode
   NROLLOUTS = NGOD * 10
   SYNCUPDATES = 100
   DEVICE = 'cpu:0'

   BATCH = 32
   SAMPLE = False
   NATTN = 2
   NPOP = 1
   SHAREINIT = False
   ENTROPY = 0.01
   VAMPYR = 1
   AUTO_TARGET = False

#Foraging only
class Law(Experiment):
   pass

#Foraging + Combat
class Chaos(Experiment):
   def vamp(self, ent, targ, frac, dmg):
      dmg = int(frac * dmg)
      targ.food.decrement(amt=dmg)
      targ.water.decrement(amt=dmg)
      ent.food.increment(amt=dmg)
      ent.water.increment(amt=dmg)

   #Damage formulas. Lambdas don't pickle well
   def MELEEDAMAGE(self, ent, targ):
      dmg = 10
      targ.applyDamage(dmg)
      self.vamp(ent, targ, self.VAMPYR, dmg)
      return dmg

   def RANGEDAMAGE(self, ent, targ):
      dmg = 2
      targ.applyDamage(dmg)
      self.vamp(ent, targ, self.VAMPYR, dmg)
      return dmg

   def MAGEDAMAGE(self, ent, targ):
      dmg = 1
      targ.applyDamage(dmg)
      self.vamp(ent, targ, self.VAMPYR, dmg)
      return dmg
