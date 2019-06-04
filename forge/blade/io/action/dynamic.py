from pdb import set_trace as T
import numpy as np

from forge.blade.io import action, utils

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

      self.out = {}
      self.nxt = None
      self.ret = ActionArgs()

   @property
   def atnArgs(self):
      return self.ret

   @property
   def outs(self):
      return self.out
    
   def next(self, env, ent, atn, outs=None):
      done = False
      if outs is not None:
         self.out[atn] = outs

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
      for _, out in outs.items():
         arguments, idx = out
         args, idx = [], int(out[1])
         for e in arguments:
            #May need e.serial[-1]
            #to form a unique key
            args.append(e.serial)
         ret.append([args, idx])
      return ret

   #Dimension packing: batch, atnList, atn (may be an action itself)
   def batch(actionLists):
      atnTensor, idxTensor, lenTensor = [], [], []

      #Pack inner set
      for actionList in actionLists:
         atns, idxs = [], []
         for atn, idx in actionList:
            atns.append(np.array(atn))
            idxs.append(idx)
         
         idxs = np.array(idxs)
         atns, lens = utils.pack(atns)

         atnTensor.append(atns)
         idxTensor.append(idxs)
         lenTensor.append(lens)

      #Pack outer set
      idxTensor, _ = utils.pack(idxTensor)
      atnTensor, _ = utils.pack(atnTensor)
      lenTensor    = utils.pack(lenTensor)

      return atnTensor, idxTensor, lenTensor

   def unbatch(atnTensor, idxTensor, lenTensor):
      lenTensor, lenLens = lenTensor
      actions = []

      #Unpack outer set (careful with unpack dim)
      atnTensor = utils.unpack(atnTensor, lenLens, dim=1)
      idxTensor = utils.unpack(idxTensor, lenLens, dim=1)
      lenTensor = utils.unpack(lenTensor, lenLens, dim=1)

      #Unpack inner set
      for atns, idxs, lens in zip(atnTensor, idxTensor, lenTensor):
         atns = utils.unpack(atns, lens)
         actions.append(list(zip(atns, idxs)))

      return actions
 
class Arg:
   def __init__(self, val, discrete=True, set=False, min=-1, max=1):
      self.val = val
      self.discrete = discrete
      self.continuous = not discrete
      self.min = min
      self.max = max
      self.n = self.max - self.min + 1

