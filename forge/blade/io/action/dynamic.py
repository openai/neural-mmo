from pdb import set_trace as T
import numpy as np

from forge.blade.io import action, utils

class ActionArgs:
   def __init__(self, action=None, args=None):
      self.action = action
      self.args = args

#ActionTree
class Dynamic:
   def __init__(self, world, entity, rootVersion, config):
      self.world, self.entity = world, entity
      self.config = config

      self.root = rootVersion
      self.out = {}
      self.nxt = None
      self.ret = ActionArgs()

   #Dimension packing: batch, atnList, atn (may be an action itself)
   def batch(actionLists):
      atnTensor = []
      idxTensor = []
      lenTensor = []

      for actionList in actionLists:
         atns, idxs = [], []
         for atn in actionList:
            atn, idx = atn

            atns.append(np.array(atn))
            idxs.append(idx)
         
         atns, lens = utils.pack(atns)
         atnTensor.append(atns)
         idxTensor.append(np.array(idxs))
         lenTensor.append(lens)

      idxTensor = utils.pack(idxTensor)
      atnTensor = utils.pack(atnTensor)
      lenTensor = utils.pack(lenTensor)

      return atnTensor, idxTensor, lenTensor

   def unbatch(atnTensor, idxTensor, lenTensor):
      atnTensor, atnLens = atnTensor
      idxTensor, idxLens = idxTensor
      lenTensor, lenLens = lenTensor

      #Careful with unpack dim here
      atnTensor = utils.unpack(atnTensor, atnLens, dim=1)
      idxTensor = utils.unpack(idxTensor, idxLens, dim=1)
      lenTensor = utils.unpack(lenTensor, lenLens, dim=1)

      actionLists = []
      for atns, idxs, lens in zip(atnTensor, idxTensor, lenTensor):
         #fix this before here
         atns = atns.astype(np.int)
         idxs = idxs.astype(np.int)
         lens = lens.astype(np.int) 
         for i, l in zip(idxs, lens):
            assert i < l
      
         atns = utils.unpack(atns, lens)
         actions = list(zip(atns, idxs))
         actionLists.append(actions)

      return actionLists
      
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

   @property
   def atnArgs(self):
      return self.ret

   @property
   def outs(self):
      return self.out

   @staticmethod
   def flat(root=action.Static):
      rets = [root]
      if root.nodeType is action.NodeType.SELECTION:
         for edge in root.edges:
            rets += action.Dynamic.flat(edge)
      return rets

class Arg:
   def __init__(self, val, discrete=True, set=False, min=-1, max=1):
      self.val = val
      self.discrete = discrete
      self.continuous = not discrete
      self.min = min
      self.max = max
      self.n = self.max - self.min + 1

