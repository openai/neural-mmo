from pdb import set_trace as T

import scipy.stats as stats
from matplotlib import pyplot as plt
import numpy as np

import os

import vec_noise
from imageio import imread, imsave
from tqdm import tqdm

from neural_mmo.forge.blade.lib import material

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
      path = self.config.PATH_TILE
      for mat in material.All:
         key = mat.tex
         tex = imread(path.format(key))
         mat.tex = tex[:, :, :3][::4, ::4]
         mat.tex = mat.tex.reshape(-1, 3).mean(0).astype(np.uint8)
         lookup[mat.index] = mat.tex.reshape(1, 1, 3)
         #lookup[mat.index] = mat.tex
         setattr(Terrain, key.upper(), mat.index)
      self.textures = lookup

   def material(self, config, val):
      if val <= config.TERRAIN_WATER:
         return Terrain.WATER
      if val <= config.TERRAIN_GRASS:
         return Terrain.GRASS
      if val <= config.TERRAIN_FOREST:
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

      #Train and eval map indices
      msg    = 'Generating {} training and {} evaluation maps:'
      evalMaps  = range(-config.TERRAIN_EVAL_MAPS, 0)
      trainMaps = range(1, config.TERRAIN_TRAIN_MAPS+1)
      print(msg.format(config.TERRAIN_TRAIN_MAPS, config.TERRAIN_EVAL_MAPS))

      llen = config.TERRAIN_EVAL_MAPS + config.TERRAIN_TRAIN_MAPS
      perm = np.random.RandomState(seed=0).permutation(llen)
      interpolaters = np.logspace(config.TERRAIN_LOG_INTERPOLATE_MIN,
                                  config.TERRAIN_LOG_INTERPOLATE_MAX,
                                  llen)[perm]

      for idx, seed in enumerate(tqdm([*evalMaps, *trainMaps])):
         path = prefix + '/map' + str(seed)
         mkdir(prefix)
         mkdir(path)

         terrain, tiles = self.grid(config, seed, interpolaters[idx])

         Save.np(tiles, path)
         if config.TERRAIN_RENDER:
            Save.fractal(terrain, path+'/fractal.png')
            Save.render(tiles, self.textures, path+'/map.png')

   def grid(self, config, seed, interpolate):
      center      = config.TERRAIN_CENTER
      border      = config.TERRAIN_BORDER
      size        = config.TERRAIN_SIZE
      frequency   = config.TERRAIN_FREQUENCY
      offset      = config.TERRAIN_FREQUENCY_OFFSET
      octaves     = center // config.TERRAIN_TILES_PER_OCTAVE

      #Data buffers
      val   = np.zeros((size, size, octaves))
      scale = np.zeros((size, size, octaves))
      s     = np.arange(size)
      X, Y  = np.meshgrid(s, s)

      #Compute noise over logscaled octaves
      start = frequency
      end   = min(start, start - np.log2(center) + offset)
      for idx, freq in enumerate(np.logspace(start, end, octaves, base=2)):
         val[:, :, idx] = vec_noise.snoise2(seed*size + freq*X, idx*size + freq*Y)

      #Compute L1 distance
      x      = np.abs(np.arange(size) - size//2)
      X, Y   = np.meshgrid(x, x)
      data   = np.stack((X, Y), -1)
      l1     = np.max(abs(data), -1)

      #Interpolation Weights
      rrange = np.linspace(-1, 1, 2*octaves-1)
      pdf    = stats.norm.pdf(rrange, 0, interpolate)
      pdf    = pdf / max(pdf)
      high   = center / 2
      delta  = high / octaves

      #Compute perlin mask
      noise  = np.zeros((size, size))
      X, Y   = np.meshgrid(s, s)
      expand = int(np.log2(center)) - 2
      for idx, octave in enumerate(range(expand, 1, -1)):
         freq, mag = 1 / 2**octave, 1 / 2**idx
         noise    += mag * vec_noise.snoise2(seed*size + freq*X, idx*size + freq*Y) 

      #plt.imshow(noise)
      #plt.show()
      noise -= np.min(noise)
      noise = octaves * noise / np.max(noise) - 1e-12
      noise = noise.astype(np.int)
      #plt.imshow(noise)
      #plt.show()

      #Compute L1 and Perlin scale factor
      for i in range(octaves):
         start             = octaves - i - 1
         scale[l1 <= high] = np.arange(start, start + octaves)
         high             -= delta

      start   = noise - 1
      l1Scale = np.clip(l1, 0, size//2 - border - 2) 
      l1Scale = l1Scale / np.max(l1Scale)
      for i in range(octaves):
         idxs           = l1Scale*scale[:, :, i] + (1-l1Scale)*(start + i)
         scale[:, :, i] = pdf[idxs.astype(np.int)]

      #Blend octaves
      std = np.std(val)
      val = val / std
      val = scale * val
      val = np.sum(scale * val, -1)
      val = std * val / np.std(val)
      val = 0.5 + np.clip(val, -1, 1)/2

      #Threshold to materials
      matl = np.zeros((size, size), dtype=object)
      for y in range(size):
         for x in range(size):
            matl[y, x] = self.material(config, val[y, x])

      #Lava and grass border
      matl[l1 > size/2 - border]       = Terrain.LAVA
      matl[l1 == size//2 - border]     = Terrain.GRASS

      edge  = l1 == size//2 - border - 1
      stone = (matl == Terrain.STONE) | (matl == Terrain.WATER)
      matl[edge & stone] = Terrain.FOREST

      return val, matl
