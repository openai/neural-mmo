from pdb import set_trace as T
import numpy as np

from forge.blade.io import action, utils
from forge.blade.io.serial import Serial

def reverse(f):
    return f.__class__(map(reversed, f.items()))

class ActionArgs:
   def __init__(self, action=None, args=None):
      self.action = action
      self.args = args

#ActionTree
class Dynamic:
   def __init__(self, world, entity, config):
      self.world, self.entity = world, entity
      self.config = config

      self.nxt = None
      self.ret = ActionArgs()
      self.prev = None

      self.out = {}

   @property
   def atnArgs(self):
      return self.ret

   @property
   def outs(self):
      return self.out
   
   def next(self, env, ent, atn, outs=None):
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

         return [args], done
 
      args = atn.args(env, ent, self.config)
      if atn.nodeType is action.NodeType.ACTION:
         self.ret.action = atn
         self.nxt = args
         done = len(args) == 0

      return args, done

   @staticmethod
   def flat(root=action.Static):
      rets = [root]
      if root.nodeType is action.NodeType.SELECTION:
         for edge in root.edges:
            rets += action.Dynamic.flat(edge)
      return rets

   @staticmethod
   def leaves(root=action.Static):
      rets = []
      for e in Dynamic.flat():
         if e.leaf:
            rets.append(e)
      return rets

   @staticmethod
   def actions(root=action.Static):
      rets = []
      for e in Dynamic.flat():
         if e.nodeType is action.NodeType.SELECTION:
            rets.append(e)
      return rets

   def serialize(outs, iden):
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

   #Dimension packing: batch, atnList, atn (may be an action itself), serial key
   def batch(actionLists):
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
 
class Arg:
   def __init__(self, val, discrete=True, set=False, min=-1, max=1):
      self.val = val
      self.discrete = discrete
      self.continuous = not discrete
      self.min = min
      self.max = max
      self.n = self.max - self.min + 1

