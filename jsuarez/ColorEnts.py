from pdb import set_trace as T
import numpy as np
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
from forge.blade.lib.enums import makeColor

fDir = 'resource/Entity/'
fName = 'neural'
extension = '.png'
neural = imread('resource/docs/' + fName + extension)
#neural = np.concatenate((neural, 255+np.zeros((16, 16, 1))), axis=2)

R, C, three = neural.shape
for r in range(R):
    for c in range(C):
        if neural[r,c,0] == 214:
           neural[r, c, -1] = 0

inds = [(5, 13), (5, 14), (6, 13), (6, 14), 
        (9, 9), (9, 10), (10, 9), (10, 10), 
        (13, 13), (13, 14), (14, 13), (14, 14)]

parh, parv = np.meshgrid(np.linspace(0.075, 1, 16), np.linspace(0.25, 1, 16)[::-1])
parh, parv = parh.T.ravel(), parv.T.ravel()

idxs = np.arange(256)
params = zip(idxs, parh, parv)
colors = [makeColor(idx, h=h, s=1, v=v) for idx, h, v in params]
for color in colors:
   name, val = color.name, color.value
   for r, c in inds:
      neural[r, c, :3] = val
   f = fDir + fName + name + extension
   imsave(f, neural.astype(np.uint8))
