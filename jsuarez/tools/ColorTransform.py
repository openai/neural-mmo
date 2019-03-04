from pdb import set_trace as T
import numpy as np
from scipy.misc import imread, imsave
from sim.lib.Enums import Neon
from skimage.color import rgb2lab, deltaE_cie76

fdir = 'resource/Material/'
fname = 'textures.png'
tex = imread(fdir+fname)
colors = (
        Neon.RED, Neon.ORANGE, Neon.YELLOW,
        Neon.GREEN, Neon.MINT, Neon.CYAN,
        Neon.BLUE, Neon.PURPLE, Neon.MAGENTA,
        Neon.FUCHSIA, Neon.SPRING, Neon.SKY,
        Neon.BLOOD, Neon.BROWN, Neon.GOLD, Neon.SILVER)
colors = np.stack([Neon.rgb(c) for c in colors])
sz = tex.shape[0]
alpha = tex[:, :, 3]
tex = tex[:, :, :3]

tex = tex.reshape(-1, 1, 3)
tex = rgb2lab(tex/255)

clrs = colors.reshape(1, -1, 3)
clrs = rgb2lab(clrs/255)

dists = deltaE_cie76(tex, clrs)
#dists = np.sum((tex - clrs)**2, 2)
inds = np.argmin(dists, 1)
px = np.array([colors[i] for i in inds])
px = px.reshape(sz, sz, 3)
imsave('tex.png', px)

