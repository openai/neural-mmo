from pdb import set_trace as T

from enum import Enum, auto
from forge.blade.lib.utils import staticproperty

class NodeType(Enum):
   #Tree edges
   STATIC = auto()    #Traverses all edges without decisions 
   SELECTION = auto() #Picks an edge to follow

   #Executable actions
   ACTION    = auto() #No arguments
   CONSTANT  = auto() #Constant argument
   VARIABLE  = auto() #Variable argument

class Node:
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

   def args(stim, entity, config):
      return []
