from pdb import set_trace as T
import ray
import pickle
import time

from forge.blade.core import realm
from forge.trinity.timed import Timed, runtime, waittime

#Agent logic
class Sword(Timed):
   '''A simple Core level interface for generic, 
   persistent, and asynchronous computation over
   a colocated Neural MMO environment (Realm)

   Args:
      trinity: A Trinity object
      config: A forge.blade.core.Config object
      args: Hook for additional user arguments
      idx: An index specifying a game map file
   '''
   def __init__(self, trinity, config, args, idx):
      super().__init__()
      self.disciples = []

   @runtime
   def step(self, *args):
      '''Synchronously steps the environment (Realm)

      Args:
         packet: List of actions to step
         the environment (Realm)

      Returns:
         observations: a list of observations for each agent
         rewards: the reward obtained by each agent (0 if the
         agent is still alive, -1 if it has died)
         done: None. Provided for conformity to the Gym API
         info: None. Provided for conformity to the Gym API

      Notes:
         This is the lowest hardware layer: there is
         currently no need for an asynchronous variant
      '''
      return
