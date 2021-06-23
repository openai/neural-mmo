class NaiveRandom():
   def __init__(self, cuda=False):
      #self.cpu = cpu
      self.alive = True
      self.timeAlive = 0

   def reproduce(self):
      return NaiveRandom()

   def interact(self, pc, stimuli, actions):
      self.timeAlive += 1
      return self.decide(pc, stimuli, actions)

   def death(self):
      self.alive = False

   def decide(self, pc, stimuli, actionTree):
      done = False
      while not actionTree.isLeaf:
         actionTree.randomNode()

      #Note: will pass if no valid args
      if type(actionTree.action) == Actions.Reproduce:
         self.reproduce()
         actionTree.decideArgs([])
      else:
         actionTree.randomArgs()
         if actionTree.args is None:
            return actionTree.passActionArgs()

      action, args = actionTree.actionArgPair()
      #print(str(action) + '    ' + str(args))
      return action, args

class AlwaysPass():
   def __init__(self, cuda=False):
      self.alive = True
      self.timeAlive = 0

   def reproduce(self):
      return self.cpu.reproduce()

   def decide(self, pc, stimuli, actionTree):
      self.timeAlive += 1
      return Actions.Pass(), []

   def death(self):
      self.alive = False

