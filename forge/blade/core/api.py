class Realm:
   '''This module is a stub reference to the core Neural MMO environment
   (Realm) that documents the external API. NMMO conforms to the OpenAI Gym
   API function signatures, but argument/return contents are different in
   order to support additional environment features (e.g. persistence,
   multi/variable agent populations, hierarchical observation/action spaces,
   etc.). Internal docs available at :mod:`forge.blade.core.realm`'''
   def __init__(self, config, idx=0):
      '''                                                                     
      Args:                                                                   
         config : A Config specification object                               
         args   : Hook for command line arguments                             
         idx    : Index of the map file to load                               
      '''  
      assert False, 'Import as \"from forge.blade.core import Realm\" (e.g. Forge.blade.core.realm.Realm). This class is a stub reference for documentation generation'

   def step(self, decisions):
      '''Simulates one game tick of the Neural MMO server

      Args:                                                                   
         decisions: A dictionary of entity (agent) action choices of format::

               {
                  ent_1: {
                     action_1: [arg_1, arg_2],
                     action_2: [...],
                     ...
                  },
                  ent_2: {
                     ...
                  },
                  ...
               }

            Structuring actions in this form is handled for
            you by :mod:`forge.blade.core.io.io`'
                                                                              
      Returns:                                                                
         (list, list, set, None):

         observations: 
            A list of observations for each agent of format::

               [
                  (env_1, ent_1),
                  (env_2, ent_2),
                  ...
               ]

            Where *env_i* is a local grid crop of tiles centered on the
            i\'th agent and *ent_i* is the i\'th agent. These are raw game
            object for usage with the forge.io api.

         reward: 
            A list of rewards for each agent. Rewards are either
            floating point values or *None*. A value of *None* indicates either
            that the agent spawned this game tick (timestep), hence no action
            has been taken to generate a reward, or that the agent has died.
            The value of done will specify the latter case. By default,
            *0* will be returned for all non-*None* values.

         dones: 
            A set of entIDs corresponding to agents that have died 
            during the past game tick. It is up to the user to interpret
            these as episode bounds during rollout collection, as well as
            to append an episode termination reward (e.g. -1) if desired.

         info:
            Provided only for conformity with OpenAI Gym.

      Notes:
         It is possible to specify invalid action combinations, such as
         two movements or two attacks. In this case, one will be selected
         arbitrarily from each incompatible set. This also means that it is
         possible to specify empty action sets, in which case the agent will
         still consume resources but otherwise take no action.
      '''
      pass

   def reset(self):
      '''Call exactly once upon environment initialization to obtain the
      first set of observations. Neural MMO is persistent: it does not
      provide a reset method after initialization. Episode bounds are
      instead specified by agent lifetimes. If you must experiment with
      short-lived environment instances, simply instantiate a new Realm.
      This method internally does nothing more than call self.step({}).
                                                                              
      Returns:                                                                
         (observations, rewards, dones, info) as documented by step()
      '''    
      pass

   def reward(self, entID):
      '''Specifies the environment protocol for rewarding agents. You can
      override this method to specify custom reward behavior with full
      access to the environment state via self.

      Returns:
         float or None:

         reward: 
            The reward for the actions on the previous timestep of the 
            entity identified by entID.                                      

      Notes:                                                                  
         Reward value will be None upon agent spawn (first game tick alive)
         and upon death. You will need to interpret the "done" signal
         during rollout collection (e.g. as -1).'''
      pass

   def spawn(self):
      '''Specifies the protocol for adding agents to the environment. You can
      override this method to specify custom spawning behavior with full
      access to the environment state via self.

      Returns:                                                                
         (int, int, str):

         entID:
            An integer used to uniquely identify the entity

         popID:
            An integer used to identity membership within a population

         prefix:
            The agent will be named prefix + entID

      Notes:                                                                  
         This API hook is mainly intended for population-based research. In
         particular, it allows you to define behavior that selectively
         spawns agents into particular populations based on the current game
         state -- for example, current population sizes or performance.'''
      pass
