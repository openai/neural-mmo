from pdb import set_trace as T
import numpy as np
import os

import vec_noise
from imageio import imread, imsave
from tqdm import tqdm

from forge.blade.lib import enums

def mkdir(path):
   try:
      os.mkdir(path)
   except:
      pass

def sharp(self, noise):
   return 2 * (0.5 - abs(0.5 - noise));

class Save:
   def render(mats, lookup, path):
      images = [[lookup[e] for e in l] for l in mats]
      image = np.vstack([np.hstack(e) for e in images])
      imsave(path, image)

   def fractal(terrain, path):
      frac = (256*terrain).astype(np.uint8)
      imsave(path, frac)

   def np(mats, path):
      """"Saved a map into into a tiled compatiable file given a save_path, width
       and height of the map, and 2D numpy array specifiying enums for the array"""
      mkdir(path)
      np.save(path + '/map.npy', mats.astype(np.int))

class Terrain:
   pass

class MapGenerator:
   def __init__(self, config):
      self.config = config
      self.loadTextures()

   def loadTextures(self):
      lookup = {}
      for mat in enums.Material:
         mat = mat.value
         tex = imread(
               'resource/assets/tiles/' + mat.tex + '.png')
         key = mat.tex
         mat.tex = tex[:, :, :3][::4, ::4]
         lookup[mat.index] = mat.tex
         setattr(Terrain, key.upper(), mat.index)
      self.textures = lookup

   def material(self, config, val, gamma=0):
      assert 0 <= gamma <= 1
      alpha = config.TERRAIN_ALPHA * gamma
      beta  = config.TERRAIN_ALPHA * gamma

      if val == config.TERRAIN_LAVA:
         return Terrain.LAVA
      if val <= config.TERRAIN_WATER:
         return Terrain.WATER
      if val <= config.TERRAIN_FOREST_LOW + beta:
         return Terrain.FOREST
      if val <= config.TERRAIN_GRASS + alpha:
         return Terrain.GRASS
      if val <= config.TERRAIN_FOREST_HIGH:
         return Terrain.FOREST
      return Terrain.STONE

   def generate(self):
      print('Generating {} game maps. This may take a moment'.format(self.config.NMAPS))
      for seed in tqdm(range(self.config.NMAPS)):
         if self.config.__class__.__name__ == 'SmallMap':
            prefix = self.config.TERRAIN_DIR_SMALL
         elif self.config.__class__.__name__ == 'LargeMap':
            prefix = self.config.TERRAIN_DIR_LARGE
         else:
            prefix = self.config.TERRAIN_DIR


         path = prefix + '/map' + str(seed)
         mkdir(prefix)
         mkdir(path)

         terrain, tiles = self.grid(self.config, seed)

         Save.np(tiles, path)
         if self.config.TERRAIN_RENDER:
            Save.fractal(terrain, path+'fractal.png')
            Save.render(tiles, self.textures, path+'map.png')

   def grid(self, config, seed):
      sz          = config.TERRAIN_SIZE
      frequency   = config.TERRAIN_FREQUENCY
      octaves     = config.TERRAIN_OCTAVES
      border      = config.TERRAIN_BORDER
      invert      = config.TERRAIN_INVERT
      
      val   = np.zeros((sz, sz, octaves))
      s     = np.arange(sz)
      X, Y  = np.meshgrid(s, s)

      #Compute noise over logscaled octaves
      start, end = frequency
      for idx, freq in enumerate(np.logspace(start, end, octaves, base=2)):
         val[:, :, idx] = 0.5 + 0.5*vec_noise.snoise2(seed*sz + freq*X, idx*sz + freq*Y)

      #Compute L1 and L2 distances
      x     = np.concatenate([np.arange(sz//2, 0, -1), np.arange(1, sz//2+1)])
      X, Y  = np.meshgrid(x, x)
      data  = np.stack((X, Y), -1)
      l1    = np.max(abs(data), -1)
      l2    = np.sqrt(np.sum(data**2, -1))

      #Linear octave blend mask
      if octaves > 1:
         dist  = np.linspace(0.5/octaves, 1-0.5/octaves, octaves)[None, None, :]
         norm  = 2 * l1[:, :, None] / sz 
         if invert:
            v = 1 - abs(norm - dist)
         else:
            v = 1 - abs(1 - norm - dist)

         v   = (2*octaves-1) * (v - 1) + 1
         v   = np.clip(v, 0, 1)
      
         v  /= np.sum(v, -1)[:, :, None]
         val = np.sum(v*val, -1)

      #Paint borders and center
      val[l1 > sz//2 - border]  = config.TERRAIN_LAVA 
      val[l1 == sz//2 - border] = config.TERRAIN_GRASS
      val[l2 < 6]               = config.TERRAIN_GRASS
      val[l2 < 3.5]             = config.TERRAIN_WATER

      #Clip l1
      if octaves > 1:
         l1 = 2 * l1 / sz
         l1[l1 <= 0.25] = 0 #Only spawn food near water at edges
         l1[l1 >= 0.75] = 1 #Only spawn food near rocks at middle
      else:
         l1 = 0.5 + l1*0

      #Threshold to materials
      matl = np.zeros((sz, sz), dtype=object)
      for y in range(sz):
         for x in range(sz):
            matl[y, x] = self.material(config, val[y, x], l1[y, x])
 
      return val, matl
