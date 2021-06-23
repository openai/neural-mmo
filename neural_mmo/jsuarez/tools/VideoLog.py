from pdb import set_trace as T
import cv2
import numpy as np
from scipy.misc import imresize
 
framedir = 'resource/data/'
prefixes = 'sl ml ll sr mr lr'.split()
suffixes = [str(i) for i in range(250)]
sz = 1024
fNames, frames = [], []
for suf in suffixes:
    print(suf)
    fName = framedir + 'slframe' + suf + '.png'
    sl = cv2.imread(fName, cv2.IMREAD_COLOR)[:256, :256, :]
    sl = imresize(sl, (sz, sz))

    fName = framedir + 'srframe' + suf + '.png'
    sr = cv2.imread(fName, cv2.IMREAD_COLOR)[:256, :256, :]
    sr = imresize(sr, (sz, sz))

    fName = framedir + 'mlframe' + suf + '.png'
    ml = cv2.imread(fName, cv2.IMREAD_COLOR)[:512, :512, :]
    ml = imresize(ml, (sz, sz))

    fName = framedir + 'mrframe' + suf + '.png'
    mr = cv2.imread(fName, cv2.IMREAD_COLOR)[:512, :512, :]
    mr = imresize(mr, (sz, sz))

    fName = framedir + 'llframe' + suf + '.png'
    ll = cv2.imread(fName, cv2.IMREAD_COLOR)[:1024, :1024, :]
    ll = imresize(ll, (sz, sz))

    fName = framedir + 'lrframe' + suf + '.png'
    lr = cv2.imread(fName, cv2.IMREAD_COLOR)[:1024, :1024, :]
    lr = imresize(lr, (sz, sz))

    l = np.concatenate((sl, ml, ll), axis=1)
    r = np.concatenate((sr, mr, lr), axis=1)
    frame = np.concatenate((l, r), axis=0)
    frame = np.stack((frame[:,:,2], frame[:,:,1], frame[:,:,0]), axis=2)
    frames.append(frame)

import imageio
imageio.mimwrite('godsword.mp4', frames, fps = 6)

