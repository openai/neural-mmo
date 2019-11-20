from pdb import set_trace as T

from collections import defaultdict
from forge.ethyr.io import Stimulus, Action
from forge.ethyr.io.action import ActionArgs

class Input:
   def __init__(self, ob, reward, done, config, serialize):
      if type(ob) != tuple:
         env, ent = None, ob
      else:
         env, ent = ob
 
      self.key   = (ent.annID, ent.entID)
      self.entID = ent.entID
      self.annID = ent.annID

      self.reward = reward
      self.done   = done

      self.rawAction = None
      if env is not None:
         self.stim   = Stimulus.process(env, ent, config, serialize)
         action      = Action.process(env, ent, config, serialize)

         self.rawAction, self.action = action

class Output:
   def __init__(self, key, atn, atnIdx, value):
      self.key    = key
      self.out    = (key, atnIdx, atn)

      self.action = {key: atnIdx}
      self.value  = value

class Inp:
   def __init__(self):
      self.keys    = []
      self.obs     = []
      self.rewards = []
      self.dones   = []

   def process(self, env, ent, reward, done, config, serialize):

      annID, entID = ent.annID, ent.entID
      key = (annID, entID)
      self.keys.append(key)
      self.rewards.append(reward)
      self.dones.append(done)

      #Perform batching operations from experience/ethyr
      #combining and bypassing ethyr/io
      rawAction = None
      if env is not None:
         self.stim         = Stimulus.process(env, ent, config, serialize)
         rawAction, action = Action.process(env, ent, config, serialize)

         self.rawAction, self.action = action
   
      return entID, key, rawAction


class IO:
   '''High level I/O class for abstracting game state and action selection'''
   def inputs(obs, rewards, dones, groupFn, config, serialize):
      '''Preprocess inputs'''
      inputs = defaultdict(Inp)
      actions    = []

      #1. figure out what data postprocessing needs from obs
      #2. Incorporate everyting into a single batch inp process operation
      for ob, reward, done in zip(obs, rewards, dones):
         if type(ob) != tuple:
            env, ent = None, ob
         else:
            env, ent = ob
    
         idx = groupFn(ent)
         entID, key, rawAction = inputs[idx].process(
               env, ent, reward, done, config, serialize)

         #Do not pass raw action to the client
         actions.append((entID, key, done, rawAction))

      return inputs, actions

   def outputs(inputs, atns):
      '''Postprocess outputs'''
      actions = defaultdict(list)
      for entID, key, done, rawAtn in inputs:
         if done:
            continue

         entAtns      = atns[key]

         #This relies on dict ordering in python 3.7+
         for atnArgs, entAtn in zip(rawAtn.items(), entAtns):
            atn, args = atnArgs
            args      = args[entAtn]

            actions[entID].append(args)

      return actions

   def batch(obs):
      stims   = Stimulus.batch([ob.stim for ob in obs])
      actions = Action.batchInputs([ob.action for ob in obs])

      return stims, actions
