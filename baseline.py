from pdb import set_trace as T
import numpy as np
import random

import projekt


from forge.blade.systems import combat, equipment, ai
from forge.blade.io.action import static as Action
from forge.trinity.twistedserver import Application
from forge.blade.core.env import Env

class Baseline:
   def __init__(self):
      config   = projekt.Config()
      self.env = Env(config)
      self.obs = self.env.reset()

      self.lifetimes = []
      t = 0
      while True:
         self.tick(None, None)
         print('Tick: ', t)
         t += 1
         if len(self.lifetimes) > 10:
            break
      perf = np.mean(self.lifetimes)
      print('Performance: ', perf)

      #Application(self.env, self.tick)

   def tick(self, pos, cmd):
      realm   = self.env.realm
      actions = {}
      for agentID in self.obs:
         if self.dones[agentID]:
            continue

         agent              = realm.players[agentID]
         agent.skills.style = Action.Range

         #agent.resources.food.increment(10)
         #agent.resources.water.increment(10)
         
         actions[agentID] = ai.policy.baseline(realm, agent)

      self.obs, _, self.dones, self.infos = self.env.step(actions)
      self.env.overlayPos = pos

      for entID, e in self.infos.items():
         self.lifetimes.append(e)

if __name__ == '__main__':
   Baseline() 
