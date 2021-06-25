import inspect

from neural_mmo.forge.blade.io.stimulus import node


class IterableTypeCompare(type):
   def __iter__(cls):
      stack = list(cls.__dict__.items())
      while len(stack) > 0:
         name, attr = stack.pop()
         if type(name) != tuple:
            name = tuple([name])
         if not inspect.isclass(attr):
            continue
         if issubclass(attr, node.Flat):
            for n, a in attr.__dict__.items():
               n = name + tuple([n]) 
               stack.append((n, a))
            continue
         yield name, attr

   def values(cls):
      return [e[1] for e in cls]

   def __hash__(self):
      return hash(self.__name__)

   def __eq__(self, other):
      return self.__name__ == other.__name__

   def __ne__(self, other):
      return self.__name__ != other.__name__

   def __lt__(self, other):
      return self.__name__ < other.__name__

   def __le__(self, other):
      return self.__name__ <= other.__name__

   def __gt__(self, other):
      return self.__name__ > other.__name__

   def __ge__(self, other):
      return self.__name__ >= other.__name__

class Iterable(type):
   def __iter__(cls):
      stack = list(cls.__dict__.items())
      while len(stack) > 0:
         name, attr = stack.pop()
         if name.startswith('__'):
            continue
         yield name, attr


