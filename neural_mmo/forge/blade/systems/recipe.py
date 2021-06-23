from neural_mmo.forge.blade import lib

class Recipe:
   def __init__(self, *args, amtMade=1):
      self.amtMade = amtMade 
      self.blueprint = lib.MultiSet()
      for i in range(0, len(args), 2):
         inp = args[i]
         amt = args[i+1]
         self.blueprint.add(inp, amt)
