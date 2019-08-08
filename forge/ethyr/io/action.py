from pdb import set_trace as T
import numpy as np

from forge.blade.io import Action as Static
from forge.blade.io.action.node import NodeType
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

   @property
   def atnArgs(self):
      '''Final chosed action argument pair'''
      return self.ret

   @property
   def outs(self):
      '''The logit packets from intermediate selection'''
      return self.out
   
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

   @staticmethod
   def flat(root=Static):
      '''Returns a flat action tree'''
      rets = [root]
      if root.nodeType is NodeType.SELECTION:
         for edge in root.edges:
            rets += Action.flat(edge)
      return rets

   @staticmethod
   def leaves(root=Static):
      '''Returns only the action leaves'''
      rets = []
      for e in Action.flat():
         if e.leaf:
            rets.append(e)
      return rets

   @staticmethod
   def actions(root=Static):
      '''Returns only selection nodes'''
      rets = []
      for e in Action.flat():
         if e.nodeType is action.NodeType.SELECTION:
            rets.append(e)
      return rets

   def serialize(outs, iden):
      '''Internal action serializer for communication across machines'''
      from forge.ethyr.io.serial import Serial
      ret = []
      for key, out in outs.items():
         key = Serial.key(key, iden)

         arguments, idx = out
         args, idx = [], int(out[1])
         for e in arguments:
            #May need e.serial[-1]
            #to form a unique key
            k = Serial.key(e, iden)
            args.append(k)
        
         ret.append([key, args, idx])
      return ret

   #Dimension packing: batch, atnList, atn, serial key
   def batch(actionLists):
      '''Internal batcher for lists of actions'''
      atnTensor, idxTensor = [], []
      keyTensor, lenTensor = [], []

      #Pack inner set
      for actionList in actionLists:
         keys, atns, idxs = [], [], []
         for key, atn, idx in actionList:
            atns.append(np.array(atn))
            idxs.append(idx)
            keys.append(key)
         
         idxs = np.array(idxs)
         keys = np.array(keys)
         atns, lens = utils.pack(atns)

         atnTensor.append(atns)
         idxTensor.append(idxs)
         keyTensor.append(keys)
         lenTensor.append(lens)

      #Pack outer set
      idxTensor, _ = utils.pack(idxTensor)
      atnTensor, _ = utils.pack(atnTensor)
      keyTensor, _ = utils.pack(keyTensor)
      lenTensor    = utils.pack(lenTensor)

      return atnTensor, idxTensor, keyTensor, lenTensor

   def unbatch(atnTensor, idxTensor, keyTensor, lenTensor):
      '''Internal inverse batcher'''
      lenTensor, lenLens = lenTensor
      actions = []

      #Unpack outer set (careful with unpack dim)
      atnTensor = utils.unpack(atnTensor, lenLens, dim=1)
      idxTensor = utils.unpack(idxTensor, lenLens, dim=1)
      keyTensor = utils.unpack(keyTensor, lenLens, dim=1)
      lenTensor = utils.unpack(lenTensor, lenLens, dim=1)

      #Unpack inner set
      for atns, idxs, keys, lens in zip(
               atnTensor, idxTensor, keyTensor, lenTensor):
         atns = utils.unpack(atns, lens, dim=-2)
         actions.append(list(zip(keys, atns, idxs)))

      return actions
