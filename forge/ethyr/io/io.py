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

class IO:
   '''High level I/O class for abstracting game state and action selection'''
   def inputs(obs, rewards, dones, config, serialize):
      '''Preprocess inputs'''
      rets    = []
      rawAtns = []
      
      for ob, reward, done in zip(obs, rewards, dones):
         val = Input(ob, reward, done, config, serialize)
         rawAtns.append(val.rawAction)

         #Do not pass this to the client
         val.rawAction = None
         rets.append(val)

      return rets, rawAtns

   def outputs(obs, rawAtns, atns):
      '''Postprocess outputs'''
      actions = defaultdict(list)
      for ob, rawAtn in zip(obs, rawAtns):
         if not hasattr(ob, 'stim'):
            continue

         entID, key   = ob.entID, ob.key
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
