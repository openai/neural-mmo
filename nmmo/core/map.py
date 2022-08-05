from pdb import set_trace as T
import numpy as np
import logging

from nmmo import core
from nmmo.lib import material

import os

class Map:
   '''Map object representing a list of tiles
   
   Also tracks a sparse list of tile updates
   '''
   def __init__(self, config, realm):
      self.config = config
      self._repr  = None
      self.realm  = realm

      sz          = config.MAP_SIZE
      self.tiles  = np.zeros((sz, sz), dtype=object)

      for r in range(sz):
         for c in range(sz):
            self.tiles[r, c] = core.Tile(config, realm, r, c)

   @property
   def packet(self):
       '''Packet of degenerate resource states'''
       missingResources = []
       for e in self.updateList:
           missingResources.append(e.pos)
       return missingResources

   @property
   def repr(self):
      '''Flat matrix of tile material indices'''
      if not self._repr:
          self._repr = [[t.mat.index for t in row] for row in self.tiles]

      return self._repr

   def reset(self, realm, idx):
      '''Reuse the current tile objects to load a new map'''
      config = self.config
      self.updateList = set()

      path_map_suffix = config.PATH_MAP_SUFFIX.format(idx)
      fPath = os.path.join(config.PATH_CWD, config.PATH_MAPS, path_map_suffix)

      try:
         map_file = np.load(fPath)
      except FileNotFoundError:
         print('Maps not found')
         raise

      materials = {mat.index: mat for mat in material.All}
      for r, row in enumerate(map_file):
         for c, idx in enumerate(row):
            mat  = materials[idx]
            tile = self.tiles[r, c]
            tile.reset(mat, config)

   def step(self):
      '''Evaluate updatable tiles'''
      if self.config.LOG_MILESTONES and self.realm.quill.milestone.log_max(f'Resource_Depleted', len(self.updateList)) and self.config.LOG_VERBOSE:
         logging.info(f'RESOURCE: Depleted {len(self.updateList)} resource tiles')                           


      for e in self.updateList.copy():
         if not e.depleted:
            self.updateList.remove(e)
         e.step()

   def harvest(self, r, c, deplete=True):
      '''Called by actions that harvest a resource tile'''
      if deplete:
          self.updateList.add(self.tiles[r, c])

      return self.tiles[r, c].harvest(deplete)
