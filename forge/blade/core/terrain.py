from pdb import set_trace as T
import numpy as np
import os

import vec_noise
from imageio import imread, imsave
from tqdm import tqdm

from forge.blade.lib import material

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
      '''Saves a map into into a tiled compatiable file given a save_path, width
       and height of the map, and 2D numpy array specifiying enums for the array'''
      mkdir(path)
      path = os.path.join(path, 'map.npy')
      np.save(path, mats.astype(np.int))

class Terrain:
   pass

class MapGenerator:
   def __init__(self, config):
      self.config = config
      self.loadTextures()

   def loadTextures(self):
      lookup = {}
      for mat in material.All:
         tex = imread('resource/assets/tiles/' + mat.tex + '.png')
         key = mat.tex
         mat.tex = tex[:, :, :3][::4, ::4]
         lookup[mat.index] = mat.tex
         setattr(Terrain, key.upper(), mat.index)
      self.textures = lookup

   def material(self, config, val, gamma=0):
      assert 0 <= gamma <= 1
      alpha = (1 - gamma) * config.TERRAIN_ALPHA
      beta  = config.TERRAIN_BETA * gamma

      if val == config.TERRAIN_LAVA:
         return Terrain.LAVA
      if val <= config.TERRAIN_WATER:
         return Terrain.WATER
      if val <= config.TERRAIN_FOREST_LOW - alpha:
         return Terrain.FOREST
      if val <= config.TERRAIN_GRASS + beta:
         return Terrain.GRASS
      if val <= config.TERRAIN_FOREST_HIGH:
         return Terrain.FOREST
      return Terrain.STONE

   def generate(self):
      config = self.config
      if config.__class__.__name__ == 'SmallMaps':
         prefix = config.PATH_MAPS_SMALL
      elif config.__class__.__name__ == 'LargeMaps':
         prefix = config.PATH_MAPS_LARGE
      else:
         prefix = config.PATH_MAPS

      msg    = 'Generating {} training and {} evaluation maps:'
      maps   = range(-config.N_EVAL_MAPS, config.N_TRAIN_MAPS+1)
      print(msg.format(config.N_TRAIN_MAPS, config.N_EVAL_MAPS))
      for seed in tqdm(maps):
         if seed == 0:
            continue

         path = prefix + '/map' + str(seed)
         mkdir(prefix)
         mkdir(path)

         terrain, tiles = self.grid(config, seed)

         Save.np(tiles, path)
         if config.TERRAIN_RENDER:
            Save.fractal(terrain, path+'/fractal.png')
            Save.render(tiles, self.textures, path+'/map.png')

   def grid(self, config, seed):
      sz          = config.TERRAIN_SIZE
      frequency   = config.TERRAIN_FREQUENCY
      octaves     = config.TERRAIN_OCTAVES
      mode        = config.TERRAIN_MODE
      lerp        = config.TERRAIN_LERP
      border      = config.TERRAIN_BORDER
      waterRadius = config.TERRAIN_WATER_RADIUS
      spawnRegion = config.TERRAIN_CENTER_REGION
      spawnWidth  = config.TERRAIN_CENTER_WIDTH

      assert mode in {'expand', 'contract', 'flat'}

      val   = np.zeros((sz, sz, octaves))
      s     = np.arange(sz)
      X, Y  = np.meshgrid(s, s)

      #Compute noise over logscaled octaves
      start, end = frequency
      for idx, freq in enumerate(np.logspace(start, end, octaves, base=2)):
         val[:, :, idx] = 0.5 + 0.5*vec_noise.snoise2(seed*sz + freq*X, idx*sz + freq*Y)

      #Compute L1 and L2 distances
      x      = np.concatenate([np.arange(sz//2, 0, -1), np.arange(1, sz//2+1)])
      X, Y   = np.meshgrid(x, x)
      data   = np.stack((X, Y), -1)
      l1     = np.max(abs(data), -1)
      l2     = np.sqrt(np.sum(data**2, -1))
      thresh = l1

      #Linear octave blend mask
      if octaves > 1:
         dist  = np.linspace(0.5/octaves, 1-0.5/octaves, octaves)[None, None, :]
         norm  = 2 * l1[:, :, None] / sz 
         if mode == 'contract':
            v = 1 - abs(1 - norm - dist)
         elif mode == 'expand':
            v = 1 - abs(norm - dist)

         v   = (2*octaves-1) * (v - 1) + 1
         v   = np.clip(v, 0, 1)
      
         v  /= np.sum(v, -1)[:, :, None]
         val = np.sum(v*val, -1)
         l1  = 1 - 2*l1/sz

      #Compute distance from the edges inward
      if mode == 'contract':
         l1 = 1 - l1

      if not lerp:
         l1 = 0.5 + 0*l1

      #Threshold to materials
      matl = np.zeros((sz, sz), dtype=object)
      for y in range(sz):
         for x in range(sz):
            matl[y, x] = self.material(config, val[y, x], l1[y, x])

      #Lava border and center crop
      matl[thresh > sz//2 - border] = Terrain.LAVA

      #Grass border or center spawn region
      if mode == 'expand':
         matl[thresh <= spawnRegion]              = Terrain.GRASS
         matl[thresh <= spawnRegion-spawnWidth]   = Terrain.STONE
         matl[thresh <= spawnRegion-spawnWidth-1] = Terrain.WATER
      elif mode == 'contract':
         matl[thresh == sz//2 - border] = Terrain.GRASS
         matl[l2 < waterRadius + 1]     = Terrain.GRASS
         matl[l2 < waterRadius]         = Terrain.WATER

      return val, matl
