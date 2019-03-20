from pdb import set_trace as T

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
      self.root, self.args = rootVersion(), None
      self.stack = [self.root]

   @property
   def n(self):
      return sum([a.nArgs() for a in self.action.edges()])

   def flat(self):
      rets = []
      for action in self.root.edges(None, None):
         for args in action.args(None, None):
            rets.append((action, args))
      return rets

   def actions(self):
      return self.root.edges

   def next(self):
      if len(self.stack) == 0:
         return None
      atn = self.stack.pop()
      if atn.edges is not None:
         for edge in atn.edges:
            self.stack.append(edge)
      if atn.argType is None:
         return self.next()
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

