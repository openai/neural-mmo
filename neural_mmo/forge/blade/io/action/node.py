from pdb import set_trace as T

from enum import Enum, auto
from neural_mmo.forge.blade.lib.utils import staticproperty
from neural_mmo.forge.blade.io.comparable import IterableTypeCompare

class NodeType(Enum):
   #Tree edges
   STATIC = auto()    #Traverses all edges without decisions 
   SELECTION = auto() #Picks an edge to follow

   #Executable actions
   ACTION    = auto() #No arguments
   CONSTANT  = auto() #Constant argument
   VARIABLE  = auto() #Variable argument

class Node(metaclass=IterableTypeCompare):
   SERIAL = 2

   @staticproperty
   def edges():
      return []

   #Fill these in
   @staticproperty
   def priority():
      return None

   @staticproperty
   def type():
      return None

   @staticproperty
   def leaf():
      return False

   @classmethod
   def N(cls, config):
      return len(cls.edges)

   def args(stim, entity, config):
      return []
