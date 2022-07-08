from pdb import set_trace as T

import scipy.stats as stats
import numpy as np
import random

import os

import vec_noise
from imageio import imread, imsave
from tqdm import tqdm

from nmmo import material

def sharp(self, noise):
   '''Exponential noise sharpener for perlin ridges'''
   return 2 * (0.5 - abs(0.5 - noise));

class Save:
   '''Save utility for map files'''
   def render(mats, lookup, path):
      '''Render tiles to png'''
      images = [[lookup[e] for e in l] for l in mats]
      image = np.vstack([np.hstack(e) for e in images])
      imsave(path, image)

   def fractal(terrain, path):
      '''Render raw noise fractal to png'''
      frac = (256*terrain).astype(np.uint8)
      imsave(path, frac)

   def np(mats, path):
      '''Save map to .npy'''
      path = os.path.join(path, 'map.npy')
      np.save(path, mats.astype(int))

class Terrain:
   '''Terrain material class; populated at runtime'''
   pass

def generate_terrain(config, idx, interpolaters):
    center      = config.MAP_CENTER
    border      = config.MAP_BORDER
    size        = config.MAP_SIZE
    frequency   = config.TERRAIN_FREQUENCY
    offset      = config.TERRAIN_FREQUENCY_OFFSET
    octaves     = center // config.TERRAIN_TILES_PER_OCTAVE

    #Compute a unique seed based on map index
    #Flip seed used to ensure train/eval maps are different
    seed = idx + 1
    if config.TERRAIN_FLIP_SEED:
        seed = -seed

    #Log interpolation factor
    if not interpolaters:
        interpolaters = np.logspace(config.TERRAIN_LOG_INTERPOLATE_MIN,
                config.TERRAIN_LOG_INTERPOLATE_MAX, config.MAP_N)

    interpolate = interpolaters[idx]

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

    noise -= np.min(noise)
    noise = octaves * noise / np.max(noise) - 1e-12
    noise = noise.astype(int)

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
        scale[:, :, i] = pdf[idxs.astype(int)]

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
            v = val[y, x]
            if v <= config.TERRAIN_WATER:
                mat = Terrain.WATER
            elif v <= config.TERRAIN_GRASS:
                mat = Terrain.GRASS
            elif v <= config.TERRAIN_FOREST:
                mat = Terrain.FOREST
            else:
                mat = Terrain.STONE
            matl[y, x] = mat

    #Lava and grass border
    matl[l1 > size/2 - border]   = Terrain.LAVA
    matl[l1 == size//2 - border] = Terrain.GRASS

    edge  = l1 == size//2 - border - 1
    stone = (matl == Terrain.STONE) | (matl == Terrain.WATER)
    matl[edge & stone] = Terrain.FOREST

    return val, matl, interpolaters


def fish(config, tiles, mat, mmin, mmax):
    r = random.randint(mmin, mmax)
    c = random.randint(mmin, mmax)

    allow = {Terrain.GRASS}
    if (tiles[r, c] not in {Terrain.WATER} or
            (tiles[r-1, c] not in allow and tiles[r+1, c] not in allow and
            tiles[r, c-1] not in allow and tiles[r, c+1] not in allow)):
        fish(config, tiles, mat, mmin, mmax)
    else:
        tiles[r, c] = mat

def uniform(config, tiles, mat, mmin, mmax):
    r = random.randint(mmin, mmax)
    c = random.randint(mmin, mmax)
    
    if tiles[r, c] not in {Terrain.GRASS}:
        uniform(config, tiles, mat, mmin, mmax)
    else:
        tiles[r, c] = mat

def cluster(config, tiles, mat, mmin, mmax):
    mmin = mmin + 1
    mmax = mmax - 1

    r = random.randint(mmin, mmax)
    c = random.randint(mmin, mmax)

    matls = {Terrain.GRASS}
    if tiles[r, c] not in matls:
        return cluster(config, tiles, mat, mmin-1, mmax+1)

    tiles[r, c] = mat
    if tiles[r-1, c] in matls:
        tiles[r-1, c] = mat
    if tiles[r+1, c] in matls:
        tiles[r+1, c] = mat
    if tiles[r, c-1] in matls:
        tiles[r, c-1] = mat
    if tiles[r, c+1] in matls:
        tiles[r, c+1] = mat

def spawn_profession_resources(config, tiles):
    mmin = config.MAP_BORDER + 1
    mmax = config.MAP_SIZE - config.MAP_BORDER - 1

    for _ in range(config.PROGRESSION_SPAWN_CLUSTERS):
        cluster(config, tiles, Terrain.ORE, mmin, mmax)
        cluster(config, tiles, Terrain.TREE, mmin, mmax)
        cluster(config, tiles, Terrain.CRYSTAL, mmin, mmax)

    for _ in range(config.PROGRESSION_SPAWN_UNIFORMS):
        uniform(config, tiles, Terrain.HERB, mmin, mmax)
        fish(config, tiles, Terrain.FISH, mmin, mmax)

class MapGenerator:
   '''Procedural map generation'''
   def __init__(self, config):
      self.config = config
      self.loadTextures()

   def loadTextures(self):
      '''Called during setup; loads and resizes tile pngs'''
      lookup = {}
      path   = self.config.PATH_TILE
      scale  = self.config.MAP_PREVIEW_DOWNSCALE
      for mat in material.All:
         key = mat.tex
         tex = imread(path.format(key))
         lookup[mat.index] = tex[:, :, :3][::scale, ::scale]
         setattr(Terrain, key.upper(), mat.index)
      self.textures = lookup

   def generate_all_maps(self):
      '''Generates NMAPS maps according to generate_map

      Provides additional utilities for saving to .npy and rendering png previews'''

      config = self.config

      #Only generate if maps are not cached
      path_maps = os.path.join(config.PATH_CWD, config.PATH_MAPS)
      os.makedirs(path_maps, exist_ok=True)
      if not config.MAP_FORCE_GENERATION and os.listdir(path_maps):
          return

      if __debug__:
          print('Generating {} maps'.format(config.MAP_N))

      for idx in tqdm(range(config.MAP_N)):
         path = path_maps + '/map' + str(idx+1)
         os.makedirs(path, exist_ok=True)

         terrain, tiles = self.generate_map(idx)


         #Save/render
         Save.np(tiles, path)
         if config.MAP_GENERATE_PREVIEWS:
            b = config.MAP_BORDER
            tiles = [e[b:-b+1] for e in tiles][b:-b+1]
            Save.fractal(terrain, path+'/fractal.png')
            Save.render(tiles, self.textures, path+'/map.png')

   def generate_map(self, idx):
      '''Generate a single map

      The default method is a relatively complex multiscale perlin noise method.
      This is not just standard multioctave noise -- we are seeding multioctave noise
      itself with perlin noise to create localized deviations in scale, plus additional
      biasing to decrease terrain frequency towards the center of the map

      We found that this creates more visually interesting terrain and more deviation in
      required planning horizon across different parts of the map. This is by no means a
      gold-standard: you are free to override this method and create customized terrain
      generation more suitable for your application. Simply pass MAP_GENERATOR=YourMapGenClass
      as a config argument.'''
      config = self.config
      if config.TERRAIN_SYSTEM_ENABLED:
         if not hasattr(self, 'interpolaters'):
             self.interpolaters = None
         terrain, tiles, interpolaters = generate_terrain(config, idx, self.interpolaters)
      else:
         size    = config.MAP_SIZE
         terrain = np.zeros((size, size))
         tiles   = np.zeros((size, size), dtype=object)

         for r in range(size):
             for c in range(size):
                 linf = max(abs(r - size//2), abs(c - size//2))
                 if linf <= size//2 - config.MAP_BORDER:
                     tiles[r, c] = Terrain.GRASS
                 else:
                     tiles[r, c] = Terrain.LAVA

      if config.PROFESSION_SYSTEM_ENABLED:
         spawn_profession_resources(config, tiles)

      return terrain, tiles
