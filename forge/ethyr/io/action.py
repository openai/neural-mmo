from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io.action import static
from forge.blade.io import Action as Static
from forge.blade.io.action.node import NodeType, Node
from forge.ethyr.io.serial import Serial
from forge.ethyr.io import utils

class ActionArgs:
   '''An action argument pair'''
   def __init__(self, action=None, args=None):
      self.action = action
      self.args = args

class Action:
   '''IO class used for interacting with game actions

   Used via .next to turn the complex action tree into
   a flat list of action and argument selections.
   '''
   def __init__(self, world, entity, config):
      self.world, self.entity = world, entity
      self.config = config

      self.nxt = None
      self.ret = ActionArgs()
      self.prev = None

      self.out = {}

   def process(inp, env, ent, config, serialize):
      actions = defaultdict(list)
      outputs = defaultdict(list)

      entKey = Serial.key(ent)
      roots  = Static.edges
      #roots  = [static.Move]

      for root in roots:
         arguments = []
         for args in root.edges:
            idxs = []
            for arg in args.args(env, ent, config):
               #Check if is a static action type
               if type(arg) is type:
                  key = Serial.key(arg)
               else:
                  key = entKey + Serial.key(arg)

               #Currently fails because inp is at the start of the emb block
               idx = inp.lookup.data[key]
               idxs.append(idx)

            inp.atn.actions[root].arguments[args].append(np.array(idxs))
         
   def next(self, env, ent, atn, outs=None):
      '''Compute the available choices for the next action

      Args:
         env: the environment observation
         ent: the entity selecting actions
         atn: the previously selected action
         outs: the logit packets from selecting the previous action

      Returns:
         args: the next arguments for selection
         done: whether we have reached the end of the selection tree
      '''
      done = False

      #Record action
      if outs is not None:
         self.out[self.prev] = outs
      self.prev = atn

      #Return argument
      if self.nxt is not None:
         args = []
         if len(self.nxt) == 0:
            done = True
            return args, done 

         args = self.nxt[0]
         self.ret.args = args #Only one arg support for now
         self.nxt = self.nxt[1:]

         return [args], True #done
 
      args = atn.args(env, ent, self.config)
      if atn.nodeType is NodeType.ACTION:
         self.ret.action = atn
         self.nxt = args
         done = len(args) == 0

      return args, done
