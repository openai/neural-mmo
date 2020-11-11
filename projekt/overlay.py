from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from forge.blade.lib import overlay
from forge.blade.systems import combat
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.entity import entity, player

class Overlay:
   def __init__(self, realm, model, trainer, config):
      self.realm     = realm
      self.model     = model
      self.trainer   = trainer
      self.config    = config

      self.R, self.C = realm.size
      self.values    = np.zeros((self.R, self.C))

   def update(self, obs):
       '''Compute per-tick updates to this overlay'''
       pass

   def register(self):
       '''Compute the overlay and register it within realm'''
       pass

class OverlayRegistry:
   def __init__(self, realm, model, trainer, config):
      '''Manager class for custom overlays'''
      self.realm    = realm

      self.overlays = {
              'counts':         Counts,
              'values':         Values,
              'globalValues':   GlobalValues,
              'attention':      Attention,
              'wilderness':     Wilderness}

      for cmd, overlay in self.overlays.items():
         self.overlays[cmd] = overlay(realm, model, trainer, config)

      self.overlays['wilderness'].init()
      self.overlays['globalValues'].init()

   def step(self, obs, pos, cmd, update=[]):
      '''Compute overlays and send to the environment'''
      #Per-tick updates
      for overlay in update:
          self.overlays[overlay].update(obs)

      if cmd in self.overlays:
          self.overlays[cmd].register(obs)

      self.realm.overlayPos = pos

class Counts(Overlay):
   def __init__(self, realm, model, trainer, config):
      super().__init__(realm, model, trainer, config)
      self.values = np.zeros((self.R, self.C, config.NPOP))

   def update(self, obs):
      '''Computes a count-based exploration map by painting
      tiles as agents walk over them'''
      for entID, agent in self.realm.realm.players.items():
         pop  = agent.base.population.val
         r, c = agent.base.pos
         self.values[r, c][pop] += 1

   def register(self, obs):
      colors    = self.realm.realm.players.palette.colors
      colors    = np.array([colors[pop].rgb
            for pop in range(self.config.NPOP)])

      colorized = self.values[:, :, :, None] * colors / 255
      colorized = np.sum(colorized, -2)
      countSum  = np.sum(self.values[:, :], -1)
      data      = overlay.norm(countSum)[..., None]

      countSum[countSum==0] = 1
      colorized = colorized * data / countSum[..., None]

      self.realm.registerOverlay(colorized)

class Values(Overlay):
   def update(self, obs):
      '''Computes a local value function by painting tiles as agents
      walk over them. This is fast and does not require additional
      network forward passes'''
      for idx, agentID in enumerate(obs):
         r, c = self.realm.realm.players[agentID].base.pos
         self.values[r, c] = float(self.model.value_function()[idx])

   def register(self, obs):
      colorized = overlay.twoTone(self.values[:, :])
      self.realm.registerOverlay(colorized)

class GlobalValues(Overlay):
   def init(self):
      '''Compute a global value function map. This requires ~6400 forward
      passes and may take up to a minute. You can disable this computation
      in the config file'''
      if self.trainer is None:
         return

      print('Computing value map...')
      values    = np.zeros(self.realm.size)
      model     = self.trainer.get_policy('policy_0').model
      obs, ents = self.realm.getValStim()

      #Compute actions to populate model value function
      BATCH_SIZE = 128
      batch = {}
      final = list(obs.keys())[-1]
      for agentID in obs:
         batch[agentID] = obs[agentID]
         if len(batch) == BATCH_SIZE or agentID == final:
            self.trainer.compute_actions(batch, state={}, policy_id='policy_0')
            for idx, agentID in enumerate(batch):
               r, c = ents[agentID].base.pos
               values[r, c] = float(self.model.value_function()[idx])
            batch = {}

      print('Value map computed')
      self.colorized = overlay.twoTone(values)

   def register(self, obs):
      self.realm.registerOverlay(self.colorized)

class Attention(Overlay):
   def register(self, obs):
      '''Computes local attentional maps with respect to each agent'''
      attentions = defaultdict(list)
      for idx, agentID in enumerate(obs):
         ent   = self.realm.realm.players[agentID]
         rad   = self.config.STIM
         r, c  = ent.pos

         tiles = self.realm.realm.map.tiles[r-rad:r+rad+1, c-rad:c+rad+1].ravel()
         for tile, a in zip(tiles, self.model.attention()[idx]):
            attentions[tile].append(float(a))

      data = np.zeros((self.R, self.C))
      tiles = self.realm.realm.map.tiles
      for r, tList in enumerate(tiles):
         for c, tile in enumerate(tList):
            if tile not in attentions:
               continue
            data[r, c] = np.mean(attentions[tile])

      colorized = overlay.twoTone(data)
      self.realm.registerOverlay(colorized)

class Wilderness(Overlay):
   def init(self):
      '''Computes the local wilderness level'''
      data = np.zeros((self.R, self.C))
      for r in range(self.R):
         for c in range(self.C):
            data[r, c] = combat.wilderness(self.config, (r, c))

      colorized = overlay.twoTone(data, preprocess='clip', invert=True, periods=5)
      self.realm.registerOverlay(colorized)
      self.wildy = colorized

   def register(self, obs):
      self.realm.registerOverlay(self.wildy)

