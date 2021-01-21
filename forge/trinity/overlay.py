from pdb import set_trace as T
import numpy as np

from collections import defaultdict
from tqdm import tqdm

from forge.blade.lib import overlay
from forge.blade.lib.enums import Neon
from forge.blade.systems import combat
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.entity import entity, player

class Overlay:
   def __init__(self, config, realm, *args):
      self.config    = config
      self.realm     = realm

      self.R, self.C = realm.size
      self.values    = np.zeros((self.R, self.C))

   def update(self, obs):
       '''Compute per-tick updates to this overlay'''
       pass

   def register(self):
       '''Compute the overlay and register it within realm'''
       pass

class OverlayRegistry:
   def __init__(self, config, realm):
      '''Manager class for custom overlays'''
      self.config = config
      self.realm  = realm

      self.overlays = {
              'counts':     Counts,
              'skills':     Skills,
              'wilderness': Wilderness}

   def init(self, *args):
      for cmd, overlay in self.overlays.items():
         self.overlays[cmd] = overlay(self.config, self.realm, *args)
      return self

   def step(self, obs, pos, cmd):
      '''Per-tick updates'''
      self.realm.overlayPos = pos
      for overlay in self.overlays.values():
          overlay.update(obs)

      if cmd in self.overlays:
          self.overlays[cmd].register(obs)

class Skills(Overlay):
   def __init__(self, config, realm, *args):
      super().__init__(config, realm)
      self.nSkills = 2

      self.values  = np.zeros((self.R, self.C, self.nSkills))

   def update(self, obs):
      '''Computes a count-based exploration map by painting
      tiles as agents walk over them'''
      for entID, agent in self.realm.realm.players.items():
         r, c = agent.base.pos

         skillLvl  = (agent.skills.fishing.level + agent.skills.hunting.level)/2.0
         combatLvl = combat.level(agent.skills)

         if skillLvl == 10 and combatLvl == 3:
            continue

         self.values[r, c, 0] = skillLvl
         self.values[r, c, 1] = combatLvl

   def register(self, obs):
      values = np.zeros((self.R, self.C, self.nSkills))
      for idx in range(self.nSkills):
         ary  = self.values[:, :, idx]
         vals = ary[ary != 0]
         mean = np.mean(vals)
         std  = np.std(vals)
         if std == 0:
            std = 1

         values[:, :, idx] = (ary - mean) / std
         values[ary == 0] = 0

      colors    = np.array([Neon.BLUE.rgb, Neon.BLOOD.rgb])
      colorized = np.zeros((self.R, self.C, 3))
      amax      = np.argmax(values, -1)

      for idx in range(self.nSkills):
         colorized[amax == idx] = colors[idx] / 255
         colorized[values[:, :, idx] == 0] = 0

      self.realm.registerOverlay(colorized)

class Counts(Overlay):
   def __init__(self, config, realm, *args):
      super().__init__(config, realm)
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
      if not hasattr(self, 'wildy'):
         print('Initializing Wilderness')
         self.init()

      self.realm.registerOverlay(self.wildy)

