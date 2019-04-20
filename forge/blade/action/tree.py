from pdb import set_trace as T
from forge.blade.action import action
from forge.blade.action.action import NodeType, staticproperty

class ActionArgs:
   def __init__(self, action, args):
      self.action = action
      self.args = args

class ActionNode:
   def __init__(self):
      pass
 
   def edges(self, world, entity):
      pass

class ArgumentNode:
   pass

class ConstDiscrete(ArgumentNode):
   pass

class VariableDiscrete(ArgumentNode):
   def __init__(self):
      self.setValue = set()

   def add(self, value):
      self.setValue.add(value)

   def toList(self):
      return list(self.setValue)

   @property
   def empty(self):
      return len(self.toList()) == 0

class ActionTree:
   def __init__(self, world, entity, rootVersion):
      self.world, self.entity = world, entity
      self.root, self.args = rootVersion, None
      self.stack = []
      self.atn = None
      self.atns = {}
      self.outs = {}

   @property
   def n(self):
      return sum([a.nArgs() for a in self.action.edges()])

   def actions(self):
      return self.root.edges

   def unpackActions(self):
      return self.atns, self.outs
      atns = []
      outs = []
      for atn, args in self.decisions.items():
         nxtAtn, args = args[0]
         atns.append(nxtAction)
         outs.append(args)
      return atns, outs

   def decide(self, atn, nxtAtn, args, outs):
      if nxtAtn.nodeType in (NodeType.ACTION, NodeType.CONSTANT, NodeType.VARIABLE):
         #self.atns[nxtAtn] = args
         self.atns[atn] = (nxtAtn, args)
      self.outs[atn] = outs

   @staticmethod
   def flat(root, rets=[]):
      if root.nodeType is NodeType.STATIC:
         for edge in root.edges:
            rets = ActionTree.flat(edge, rets)
      elif root.nodeType is NodeType.SELECTION:
         rets.append(root)
         rets += root.edges
      return rets

   def pop(self):
      if len(self.stack) > 0:
         return self.stack.pop()
      return None

   #DFS for the next action -- core and tricky function
   def next(self, atn=None, args=None, outs=None):
      if atn is None:
         atn = self.root

      #Traverse all edges
      if atn.nodeType is NodeType.STATIC:
         self.stack += atn.edges
         atn = self.pop()
         self.atn = atn
         return self.next(atn)
      #Select an edge or argument
      elif atn.nodeType is NodeType.SELECTION:
         self.outs[self.atn] = outs
      #Register no-argument action
      elif atn.nodeType is NodeType.ACTION:
         self.outs[self.atn] = outs
         self.atns[self.atn] = ActionArgs(atn, None)
         atn = self.pop()
      #Must pick an argument
      elif args is None:
         self.outs[self.atn] = outs
         return atn
      #Register action with argument
      else:
         self.outs[atn] = outs
         self.atns[self.atn] = ActionArgs(atn, args)
         atn = self.pop()

      self.atn = atn
      return atn

   def rand(self):
      nodes = self.flat()
      ind = np.random.randint(0, len(nodes))
      return nodes[ind]

   def actionArgPair(self):
      assert self.action is not None and self.args is not None
      if type(self.args) not in (list, tuple):
         self.args = [self.args]
      return type(self.action), self.args

class Arg:
   def __init__(self, val, discrete=True, set=False, min=-1, max=1):
      self.val = val
      self.discrete = discrete
      self.continuous = not discrete
      self.min = min
      self.max = max
      self.n = self.max - self.min + 1

