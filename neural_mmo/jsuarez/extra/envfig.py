from pdb import set_trace as T
from scipy.misc import imread, imresize, imsave
import numpy as np
import imageio

fracs, imgs = [], []
for i in range(6):
   mapPath     = 'resource/maps/map' + str(i) + '/map.png'
   fractalPath = 'resource/maps/map' + str(i) + '/fractal.png'

   mapImg = imread(mapPath)[256:-256, 256:-256]
   fractalImg = imread(fractalPath)[8:-8, 8:-8]
   fractalImg = imresize(fractalImg, mapImg.shape)
   fractalImg = np.stack(3*[fractalImg], 2)

   fracs.append(fractalImg)
   imgs.append(mapImg)

fracs = np.concatenate(fracs, 1)
imgs  = np.concatenate(imgs, 1)
rets  = np.concatenate((fracs, imgs), 0)
imsave('envgen.png', rets)
