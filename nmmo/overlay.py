from pdb import set_trace as T
import numpy as np

from nmmo.lib import overlay
from nmmo.lib.colors import Neon
from nmmo.systems import combat


class OverlayRegistry:
   def __init__(self, config, realm):
      '''Manager class for overlays

      Args:
          config: A Config object
          realm: An environment
      '''
      self.initialized = False

      self.config = config
      self.realm  = realm

      self.overlays = {
              'counts':     Counts,
              'skills':     Skills,
              'wilderness': Wilderness}


   def init(self, *args):
      self.initialized = True
      for cmd, overlay in self.overlays.items():
         self.overlays[cmd] = overlay(self.config, self.realm, *args)
      return self

   def step(self, obs, pos, cmd):
      '''Per-tick overlay updates

      Args:
          obs: Observation returned by the environment
          pos: Client camera focus position
          cmd: User command returned by the client
      '''
      if not self.initialized:
          self.init()

      self.realm.overlayPos = pos
      for overlay in self.overlays.values():
          overlay.update(obs)

      if cmd in self.overlays:
          self.overlays[cmd].register(obs)

class Overlay:
   '''Define a overlay for visualization in the client

   Overlays are color images of the same size as the game map.
   They are rendered over the environment with transparency and
   can be used to gain insight about agent behaviors.'''
   def __init__(self, config, realm, *args):
      '''
      Args:
          config: A Config object
          realm: An environment
      '''
      self.config     = config
      self.realm      = realm

      self.size       = config.MAP_SIZE
      self.values     = np.zeros((self.size, self.size))

   def update(self, obs):
       '''Compute per-tick updates to this overlay. Override per overlay.

       Args:
           obs: Observation returned by the environment
       '''
       pass

   def register(self):
       '''Compute the overlay and register it within realm. Override per overlay.'''
       pass

class Skills(Overlay):
   def __init__(self, config, realm, *args):
      '''Indicates whether agents specialize in foraging or combat'''
      super().__init__(config, realm)
      self.nSkills = 2

      self.values  = np.zeros((self.size, self.size, self.nSkills))

   def update(self, obs):
      '''Computes a count-based exploration map by painting
      tiles as agents walk over them'''
      for entID, agent in self.realm.realm.players.items():
         r, c = agent.base.pos

         skillLvl  = (agent.skills.food.level.val + agent.skills.water.level.val)/2.0
         combatLvl = combat.level(agent.skills)

         if skillLvl == 10 and combatLvl == 3:
            continue

         self.values[r, c, 0] = skillLvl
         self.values[r, c, 1] = combatLvl

   def register(self, obs):
      values = np.zeros((self.size, self.size, self.nSkills))
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
      colorized = np.zeros((self.size, self.size, 3))
      amax      = np.argmax(values, -1)

      for idx in range(self.nSkills):
         colorized[amax == idx] = colors[idx] / 255
         colorized[values[:, :, idx] == 0] = 0

      self.realm.register(colorized)

class Counts(Overlay):
   def __init__(self, config, realm, *args):
      super().__init__(config, realm)
      self.values = np.zeros((self.size, self.size, config.PLAYER_POLICIES))

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
            for pop in range(self.config.PLAYER_POLICIES)])

      colorized = self.values[:, :, :, None] * colors / 255
      colorized = np.sum(colorized, -2)
      countSum  = np.sum(self.values[:, :], -1)
      data      = overlay.norm(countSum)[..., None]

      countSum[countSum==0] = 1
      colorized = colorized * data / countSum[..., None]

      self.realm.register(colorized)

class Wilderness(Overlay):
   def init(self):
      '''Computes the local wilderness level'''
      data = np.zeros((self.size, self.size))
      for r in range(self.size):
         for c in range(self.size):
            data[r, c] = combat.wilderness(self.config, (r, c))

      self.wildy = overlay.twoTone(data, preprocess='clip', invert=True, periods=5)

   def register(self, obs):
      if not hasattr(self, 'wildy'):
         print('Initializing Wilderness')
         self.init()

      self.realm.register(self.wildy)
