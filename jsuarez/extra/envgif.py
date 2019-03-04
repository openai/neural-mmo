from pdb import set_trace as T
from scipy.misc import imread, imresize
import numpy as np
import imageio

frames = []
for i in range(100):
   mapPath     = 'resource/maps/map' + str(i) + '/map.png'
   fractalPath = 'resource/maps/map' + str(i) + '/fractal.png'
   print(i)

   mapImg = imread(mapPath)
   fractalImg = imread(fractalPath)
   fractalImg = imresize(fractalImg, mapImg.shape)
   fractalImg = np.stack(3*[fractalImg], 2)
   
   img = np.concatenate((fractalImg, mapImg), 1)
   frames.append(img)

imageio.mimwrite('envgen.mp4', frames, fps=1.5)



