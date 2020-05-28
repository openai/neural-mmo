from pdb import set_trace as T
import numpy as np

from collections import defaultdict

class Evaluator:
   def __init__(self, trainer, env, config):
      self.obs   = env.reset()
      self.env   = env

      self.state = {}
      self.done  = {}

      self.config   = config
      config.RENDER = True

      self.trainer  = trainer
      self.values()

   #Start a persistent Twisted environment server
   def run(self):
      from forge.embyr.twistedserver import Application
      self.tick()
      self.tick()
      self.tick()
      Application(self.env, self.tick)

   #Compute actions and overlays for a single timestep
   def tick(self):
      atns, values, attns = {}, {}, defaultdict(dict)
      model = self.trainer.get_policy('policy_0').model

      #Remove dead agents
      for agentID in self.done:
         if self.done[agentID]:
            del self.obs[agentID]
           
      #Compute batch of actions
      atns, self.state, _ = self.trainer.compute_actions(
            self.obs, state=self.state, policy_id='policy_0')

      #Update values, and attention
      for idx, agentID in enumerate(self.obs):
         ent       = self.env.desciples[agentID]

         values[agentID] = float(model.value[idx])
         tiles  = self.env.raw[agentID][('Tile',)]
         for tile, a in zip(tiles, model.attn[idx]):
            attns[ent][tile] = float(a)

      #Reformat actions
      actions = defaultdict(lambda: defaultdict(dict))
      for atn, args in atns.items():
         for arg, vals in args.items():
            for idx, agentID in enumerate(self.obs):
               actions[agentID][atn][arg] = vals[idx]

      #Step the environment
      self.obs, rewards, self.done, _ = self.env.step(actions, values, attns)

   #Compute a global value function map. This requires ~6400 forward
   #passes and a ton of environment deep copy operations, which will 
   #take several minutes. You can disable this computation in the config
   def values(self):
      values = np.zeros(self.env.size)
      if not self.config.COMPUTE_GLOBAL_VALUES:
         self.env.setGlobalValues(values)
         return

      print('Computing value map...')
      values = np.zeros(self.env.size)
      for obs, stim in self.env.getValStim():
         env, ent   = stim
         r, c       = ent.base.pos

         atn, model   = self.action(ent.entID, ent, obs, mock=True)
         values[r, c] = float(model.value)

      self.env.setGlobalValues(values)
      print('Value map computed')
