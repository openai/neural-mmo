from pdb import set_trace as T
import numpy as np
from sim.lib.Enums import Neon
from tools.Colors import hex2rgb
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt

fDir = 'resource/Entity/'
fName = 'neural'
extension = '.png'
neural = imread(fDir + fName + extension)

inds = [(5, 13), (5, 14), (6, 13), (6, 14), 
        (9, 9), (9, 10), (10, 9), (10, 10), 
        (13, 13), (13, 14), (14, 13), (14, 14)]
#px = [neural[r, c] for r, c in inds]

for color in Neon:
   if color not in (Neon.RED, Neon.ORANGE, Neon.YELLOW, Neon.GREEN, 
           Neon.MINT, Neon.CYAN, Neon.BLUE, Neon.PURPLE, Neon.MAGENTA, 
           Neon.FUCHSIA, Neon.SPRING, Neon.SKY):
      continue
   name, val = color.name, hex2rgb(color.value)

   for r, c in inds:
      neural[r, c, :] = val

   f = fDir + fName + name + extension
   imsave(f, neural)
