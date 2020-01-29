from pdb import set_trace as T
from opensimplex import OpenSimplex
gen = OpenSimplex()

from forge.blade.lib import enums
from matplotlib import pyplot as plt

import sys

try:
    from imageio import imread, imsave
except ImportError:
    try:
        from scipy.misc import imread, imsave
    except ImportError:
        print(sys.exc_info())

from shutil import copyfile
from copy import deepcopy
import numpy as np
import os

template_tiled = """<?xml version="1.0" encoding="UTF-8"?>
<map version="1.0" tiledversion="1.1.5" orientation="orthogonal" renderorder="right-down" width="{0}" height="{1}" tilewidth="128" tileheight="128" infinite="0" nextobjectid="1">
 <tileset firstgid="0" source="../../tiles.tsx"/>
 <layer name="Tile Layer 1" width="{0}" height="{1}">
  <data encoding="csv">
  {2}
</data>
 </layer>
</map>"""

def saveTiled(dat, path):
   """"Saved a map into into a tiled compatiable file given a save_path,
       width and hieght of the map, and 2D numpy array specifiying enums for the array"""
   height, width = dat.shape 
   dat = str(dat.ravel().tolist())
   dat = dat.strip('[').strip(']')
   with open(path + 'map.tmx', "w") as f:
      f.write(template_tiled.format(width, height, dat))

#Bounds checker
def inBounds(r, c, shape, border=0):
   R, C = shape
   return (
         r > border and
         c > border and
         r < R - border - 1 and
         c < C - border - 1
         )

def noise(nx, ny):
    # Rescale from -1.0:+1.0 to 0.0:1.0
    return gen.noise2d(nx, ny) / 2.0 + 0.5

def sharp(nx, ny):
    return 2 * (0.5 - abs(0.5 - noise(nx, ny)));

def perlin(nx, ny, octaves, scale=1):
   val = 0
   for mag, freq in octaves:
      val += mag * noise(scale*freq*nx, scale*freq*ny)
   return

def ridge(nx, ny, octaves, scale=1):
   val = []
   for idx, octave in enumerate(octaves):
      mag, freq = octave
      v = mag*sharp(scale*freq*nx, scale*freq*ny)
      if idx > 0:
         v *= sum(val)
      val.append(v)
   return sum(val)

def expParams(n):
   return [(0.5**i, 2**i) for i in range(n)]

def norm(x, X):
   return x/X - 0.5

def grid(X, Y, n=8, scale=1, seed=0):
   terrain = np.zeros((Y, X))
   for y in range(Y):
       for x in range(X):
           octaves = expParams(n)
           nx, ny = norm(x, X), norm(y, Y)
           terrain[x, y] = ridge(seed+nx, seed+ny, octaves, scale)
   return terrain / np.max(terrain)

def textures():
   lookup = {}
   for mat in enums.Material:
      mat = mat.value
      tex = imread(
            'resource/assets/tiles/' + mat.tex + '.png')
      key = mat.tex
      mat.tex = tex[:, :, :3][::4, ::4]
      lookup[key] = mat
   return lookup

def tile(val, offset):
   if val == 0:
      return 'lava'
   elif val < 0.3:
      return 'water'
   elif val < 0.675 - offset:
      return 'grass'
   elif val < 0.75 - offset/2:
      return 'forest'
   else:
      return 'stone'

def material(terrain, tex, X, Y, border=9):
   terrain = deepcopy(terrain).astype(object)
   for y in range(Y):
      for x in range(X):
         #Center coords
         xRel = x - sz/2
         yRel = y - sz/2
         #mag  = np.sqrt(xRel**2 + yRel**2)
         mag = max(abs(xRel), abs(yRel))

         if not inBounds(y, x, (Y, X), border-1):
            mat = 'lava'
         elif not inBounds(y, x, (Y, X), border):
            mat = 'grass'
         elif np.sqrt(xRel**2 + yRel**2) < 2.5:
            mat = 'water'
         elif mag < 6:
            mat = 'grass'
         else:
            val = float(terrain[y, x])
            norm = mag / (sz / 2)
            offset = norm
            #curve = val + 0.1*mag/sz + 0.9*(mag/sz)**3
            mat = tile(val, 0.325*offset)

         terrain[y, x] = tex[mat]

   return terrain

def render(mats, path):
   images = [[e.tex for e in l] for l in mats]
   image = np.vstack([np.hstack(e) for e in images])
   imsave(path, image)

def index(mats, path):
   inds = np.array([[e.index+1 for e in l] for l in mats])
   saveTiled(inds, path)

def fractal(terrain, path):
   frac = (256*terrain).astype(np.uint8)
   imsave(path, terrain)

nMaps, sz = 12, 128 + 16
#nMaps, sz = 1, 512 + 16
seeds = np.linspace(0, 2**32, nMaps)
scale = int(sz / 5)
root = 'resource/maps/'
tex = textures()

print('Generating {} game maps. This may take a moment'.format(nMaps))
for i, seed in enumerate(seeds):
   print('Generating map ' + str(i))
   path = root + 'procedural/map' + str(i) + '/'
   try:
      os.mkdir(path)
   except:
      pass
   terrain = grid(sz, sz, scale=scale, seed=seed)
   tiles = material(terrain, tex, sz, sz)
   fractal(terrain, path+'fractal.png')
   render(tiles, path+'map.png')
   index(tiles, path)


