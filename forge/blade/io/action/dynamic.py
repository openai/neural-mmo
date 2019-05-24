from pdb import set_trace as T

from forge.blade.io import action

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

