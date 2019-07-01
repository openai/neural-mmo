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
      self.env = realm.Realm(config, args, idx)
      self.env.spawn = self.spawn

      self.ent, self.nPop = 0, config.NPOP
      self.disciples = [self.env]

   def spawn(self):
      '''Specifies how the environment adds players

      Returns:
         entID (int), popID (int), name (str): 
         unique IDs for the entity and population, 
         as well as a name prefix for the agent 
         (the ID is appended automatically).

      Notes:
         This is useful for population based research,
         as it allows one to specify per-agent or
         per-population policies'''

      pop = hash(str(self.ent)) % self.nPop
      self.ent += 1
      return self.ent, pop, 'Neural_'

   def getEnv(self):
      '''Returns the environment. Ray does not allow
      access to remote attributes without a getter'''
      return self.env

   @runtime
   def step(self, atns):
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
      return self._step(atns)

   #This is an internal function only
   #used to separate timer decorators
   @waittime
   def _step(self, atns):
      return self.env.step(atns)
