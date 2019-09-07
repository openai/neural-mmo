from pdb import set_trace as T
import ray

from forge.trinity.ascend import Ascend, runtime, waittime

class Trinity(Ascend):
   '''Pantheon-God-Sword (Cluster-Server-Core) infra 

   Trinity is three layer distributed infra design pattern
   for generic, persistent, and asynchronous computation
   at the Cluster, Server, and Core levels. It is built from
   three user-provided Ascend subclasses, which specify
   behavior at each layer. Trinity takes advantage of
   Ascend's builtin performance logging and does not 
   interfere with pdb breakpoint debugging.
  
   To use Trinity, implement Pantheon, God, and Sword as
   Ascend subclasses to specify Cluster, Server, and 
   Core level execution. Overriding the step() function 
   allows you to perform arbitrary computation and return 
   arbitrary data to the previous layer. Calling super.step() 
   allows you to send arbitrary data and receive computation 
   results from the next layer.

   Args:
      pantheon: A subclassed Pantheon object
      god: A subclassed God object
      sword: A subclassed Sword object

   Notes:
      Ascend is the base API defining distributed infra
      layers. Trinity is simply three stacked Ascend layers.
      Ascend and Trinity are not specific computation models:
      they are design patterns for creating computation models.
      You can implement anything from MPI broadcast-reduce to 
      OpenAI's Rapid to our demo's MMO style communications using
      Ascend + Trinity with relatively little code and testing.
   ''' 
   def __init__(self, pantheon, god, sword):
      self.pantheon = pantheon
      self.god      = god
      self.sword    = sword

   def init(self, config):
      '''
      Instantiates a Pantheon object to make
      Trinity runnable. Separated from __init__
      to make Trinity usable as a stuct to hold
      Pantheon, God, and Sword subclass references

      Args:
         config: A forge.blade.core.Config object
      '''
      super().__init__(self.pantheon, 1, self, config)
      return self

   @runtime
   def step(self):
      '''Wraps Pantheon step'''
      return super().step()[0]
