from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.lib import overlay
from forge.blade.io.stimulus.static import Stimulus

class Overlays:
   def __init__(self, realm, model, trainer, config):
      '''Manager class for custom overlays'''
      self.realm   = realm
      self.model   = model
      self.trainer = trainer
      self.config  = config
      self.init    = True

      self.window   = 64
      R, C          = realm.size
      self.valueMap = np.zeros((R, C))
      self.countMap = np.zeros((R, C, config.NPOP))

   def register(self, obs, pos):
      '''Compute overlays and send to the environment'''
      self.pos = pos
      self.counts(obs)
      self.values(obs)
      self.attention(obs)

      if self.config.COMPUTE_GLOBAL_VALUES or self.init:
         self.globalValues(obs)
         self.init = False
 
   def counts(self, obs):
      '''Computes a count-based exploration map by painting
      tiles as agents walk over them'''
      for entID, agent in self.realm.desciples.items():
         pop  = agent.base.population
         r, c = agent.base.pos
         self.countMap[r, c][pop] += 1

      r, c = self.pos
      w    = self.window

      colors    = self.realm.spawner.palette.colors
      colors    = np.array([colors[pop].rgb
            for pop in range(self.config.NPOP)])

      colorized = self.countMap[r-w:r+w, c-w:c+w, :, None] * colors / 255
      colorized = np.sum(colorized, -2)

      countSum  = np.sum(self.countMap[r-w:r+w, c-w:c+w], -1)
      data      = overlay.preprocess(countSum)[..., None]

      countSum[countSum==0] = 1
      colorized = colorized * data / countSum[..., None]

      self.realm.registerOverlay(colorized, 'counts')

   def values(self, obs):
      '''Computes a local value function by painting tiles as agents
      walk over them. This is fast and does not require additional
      network forward passes'''
      for idx, agentID in enumerate(obs):
         r, c = self.realm.desciples[agentID].base.pos
         self.valueMap[r, c] = float(self.model.value_function()[idx])

      r, c = self.pos
      w    = self.window

      colorized = overlay.twoTone(self.valueMap[r-w:r+w, c-w:c+w])
      self.realm.registerOverlay(colorized, 'values')

   def globalValues(self, obs):
      '''Compute a global value function map. This requires ~6400 forward
      passes and may take up to a minute. You can disable this computation
      in the config file'''
      if not self.config.COMPUTE_GLOBAL_VALUES or not self.init:
         return
      print('Computing value map...')
      values     = np.zeros(self.realm.size)
      model      = self.trainer.get_policy('policy_0').model
      obs, stims = self.realm.getValStim()

      #Compute actions to populate model value function
      self.trainer.compute_actions(obs, state={}, policy_id='policy_0')

      for agentID in obs.keys():
         env, ent = stims[agentID]
         atn      = obs[agentID]
         r, c     = ent.base.pos

         values[r, c] = float(self.model.value_function()[agentID])

      print('Value map computed')
      colorized = overlay.twoTone(values)
      self.realm.registerOverlay(colorized, 'globalValues')

   def attention(self, obs):
      '''Computes local attentional maps with respect to each agent'''
      attentions = defaultdict(list)
      for idx, agentID in enumerate(obs):
         tiles = self.realm.raw[agentID][Stimulus.Tile]
         ent   = self.realm.desciples[agentID]
         for tile, a in zip(tiles, self.model.attention()[idx]):
            attentions[tile].append(float(a))

      r, c = self.pos
      w    = self.window

      data = np.zeros((2*w, 2*w))
      tiles = self.realm.world.env.tiles[r-w:r+w, c-w:c+w]
      for r, tList in enumerate(tiles):
         for c, tile in enumerate(tList):
            if tile not in attentions:
               continue
            data[r, c] = np.mean(attentions[tile])

      colorized = overlay.twoTone(data)
      self.realm.registerOverlay(colorized, 'attention')
