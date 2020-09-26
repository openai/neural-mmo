from pdb import set_trace as T
import numpy as np

import vec_noise
from imageio import imread, imsave
from tqdm import tqdm

from forge.blade.lib import enums

def sharp(self, noise):
   return 2 * (0.5 - abs(0.5 - noise));

class Save:
   template_tiled = """<?xml version="1.0" encoding="UTF-8"?>
<map version="1.0" tiledversion="1.1.5" orientation="orthogonal" renderorder="right-down" width="{0}" height="{1}" tilewidth="128" tileheight="128" infinite="0" nextobjectid="1">
 <tileset firstgid="0" source="../../tiles.tsx"/>
 <layer name="Tile Layer 1" width="{0}" height="{1}">
  <data encoding="csv">
  {2}
</data>
 </layer>
</map>"""
   def render(mats, lookup, path):
      images = [[lookup[e] for e in l] for l in mats]
      image = np.vstack([np.hstack(e) for e in images])
      imsave(path, image)

   def fractal(terrain, path):
      frac = (256*terrain).astype(np.uint8)
      imsave(path, frac)

   def tiled(mats, path):
      """"Saved a map into into a tiled compatiable file given a save_path, width
       and height of the map, and 2D numpy array specifiying enums for the array"""
      dat = np.array([[e+1 for e in l] for l in mats])
      height, width = dat.shape 
      dat = str(dat.ravel().tolist())
      dat = dat.strip('[').strip(']')
      with open(path + 'map.tmx', "w") as f:
         f.write(Save.template_tiled.format(width, height, dat))

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

   def material(self, val, gamma=0):
      assert gamma >= 0 and gamma <= 1
      alpha = 0.035 * gamma
      beta  = 0.05 * gamma
      if val == 0:
         return Terrain.LAVA
      if val < 0.25:
         return Terrain.WATER
      if val < 0.25+beta:
         return Terrain.FOREST
      if val < 0.715+alpha:
         return Terrain.GRASS
      if val < 0.75:
         return Terrain.FOREST
      return Terrain.STONE

   def generate(self):
      print('Generating {} game maps. This may take a moment'.format(self.config.NMAPS))
      for seed in tqdm(range(self.config.NMAPS)):
         path = self.config.TERRAIN_DIR + 'map' + str(seed) + '/'

         try:
            os.mkdir(path)
         except:
            pass

         terrain, tiles = self.grid(
               sz        = self.config.TERRAIN_SIZE,
               frequency = self.config.TERRAIN_FREQUENCY,
               octaves   = self.config.TERRAIN_OCTAVES,
               border    = self.config.TERRAIN_BORDER,
               invert    = self.config.TERRAIN_INVERT,
               seed      = seed)

         Save.tiled(tiles, path)
         if self.config.TERRAIN_RENDER:
            Save.fractal(terrain, path+'fractal.png')
            Save.render(tiles, self.textures, path+'map.png')

   def grid(self, sz, frequency, octaves, border, invert, seed):
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
      val[l1 > sz//2 - border]  = 0
      val[l1 == sz//2 - border] = 0.5
      val[l2 < 6]               = 0.5
      val[l2 < 3.5]             = 0.1

      #Clip l1
      if octaves > 1:
         l1 = 2 * l1 / sz
         l1[l1 <= 0.25] = 0
         l1[l1 >= 0.75] = 1
      else:
         l1 = 0.5 + l1*0

      #Threshold to materials
      matl = np.zeros((sz, sz), dtype=object)
      for y in range(sz):
         for x in range(sz):
            matl[y, x] = self.material(val[y, x], l1[y, x])
 
      return val, matl
