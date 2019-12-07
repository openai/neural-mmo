from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.io.action import static
from forge.blade.io import Action as Static
from forge.blade.io.action.node import NodeType, Node
from forge.ethyr.io.serial import Serial
from forge.ethyr.io import utils

class Action:
   '''Static IO class used for interacting with game actions

   The environment expects formatted action dictionaries.
   This class assembles candidate arguments for agent inference'''
   def process(inp, env, ent, config, serialize):
      '''Utility for preprocessing agent decisions
      
      Built to not require updates for new classes of actions

      Args:
         inp       : An IO object specifying observations
         env       : Local environment observation
         ent       : Local entity observation
         config    : a configuration object
         serialize : (bool) Whether to serialize the IO object data
      '''
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
               if type(arg) is type: #Static reference
                  key = Serial.key(arg)
               else: #Entity instance
                  key = entKey + Serial.key(arg)

               idx = inp.lookup.data[key]
               idxs.append(idx)

            inp.atn.actions[root].arguments[args].append(np.array(idxs))
