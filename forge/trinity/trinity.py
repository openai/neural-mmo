from pdb import set_trace as T
import ray

from forge.trinity.timed import Timed, runtime, waittime

class Trinity(Timed):
   '''Pantheon-God-Sword (Cluster-Server-Core) wrapper

   Trinity is a featherweight wrapper around the
   excellent Ray API that provides a simple interface
   for generic, persistent, and asynchronous computation
   at the Cluster, Server, and Core levels. It also 
   provides builtin performance logging and does not 
   interfere with pdb breakpoint debugging.
  
   To use Trinity, override Pantheon, God, and Sword to
   specify Cluster, Server, and Core level execution.
   Overriding the step() function allows you to perform
   arbitrary computation and return arbitrary data to
   the previous layer. Calling super.step() allows you
   to send arbitrary data and receive computation results 
   from the next layer.

   Args:
      pantheon: A subclassed Pantheon object
      god: A subclassed God object
      Sword: A subclassed Sword object

   Notes:
      Trinity is not a single computation model. It is an
      interface for creating computation models. By
      example, our demo project adopts a computation
      model similar to OpenAI Rapid. But trinity makes it
      possible to move any piece of the execution among 
      hardware layers with relatively little code and testing.
   ''' 
   def __init__(self, pantheon, god, sword):
      super().__init__()
      self.pantheon = pantheon
      self.god      = god
      self.sword    = sword

   def init(self, config, args):
      '''
      Instantiates a Pantheon object to make
      Trinity runnable. Separated from __init__
      to make Trinity usable as a stuct to hold
      Pantheon, God, and Sword subclass references

      Args:
         config: A forge.blade.core.Config object
         args: Hook for additional user arguments.
      '''
      self.base = self.pantheon(self, config, args)
      self.disciples = [self.base]

   @runtime
   def step(self):
      '''Wraps Pantheon step'''
      return self.base.step()
