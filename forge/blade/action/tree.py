class ActionNode:
   def edges(self, world, entity):
      pass

class ActionLeaf(ActionNode):
   def __call__(self, world, entity, *args):
      if type(args) == EmptyArgs:
         args = []

class Args:
   isEmpty = False
   isSet = False
   isDiscrete = False
   def __init__(self, value=None):
      self.value = value

class Arg(Args): pass

class SetArgs(Args):
   isSet = True

   def __init__(self):
      self.setValue = set()

   def add(self, value):
      self.setValue.add(value)

   def toList(self):
      return list(self.setValue)

   @property
   def empty(self):
      return len(self.toList()) == 0

class DiscreteArgs(SetArgs):
   isDiscrete = True

class EmptyArgs(Args):
   isEmpty = True

class ActionTree:
   def __init__(self, world, entity, rootVersion):
      self.world, self.entity = world, entity
      self.root, self.args = rootVersion(), None

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
      return self.root.edges(None, None)

   def rand(self):
      nodes = self.flat()
      ind = np.random.randint(0, len(nodes))
      return nodes[ind]

   def actionArgPair(self):
      assert self.action is not None and self.args is not None
      if type(self.args) not in (list, tuple):
         self.args = [self.args]
      return type(self.action), self.args

