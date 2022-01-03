from pdb import set_trace as T
import numpy as np
from scipy import signal

def norm(ary, nStd=2):
   assert type(ary) == np.ndarray, 'ary must be of type np.ndarray'
   R, C         = ary.shape
   preprocessed = np.zeros_like(ary)
   nonzero      = ary[ary!= 0]
   mean         = np.mean(nonzero)
   std          = np.std(nonzero)
   if std == 0:
      std = 1
   for r in range(R):
     for c in range(C):
        val = ary[r, c]
        if val != 0:
           val = (val - mean) / (nStd * std)
           val = np.clip(val+1, 0, 2)/2
           preprocessed[r, c] = val
   return preprocessed

def clip(ary):
   assert type(ary) == np.ndarray, 'ary must be of type np.ndarray'
   R, C         = ary.shape
   preprocessed = np.zeros_like(ary)
   nonzero      = ary[ary!= 0]
   mmin         = np.min(nonzero)
   mmag         = np.max(nonzero) - mmin
   for r in range(R):
     for c in range(C):
        val = ary[r, c]
        val = (val - mmin) / mmag
        preprocessed[r, c] = val
   return preprocessed

def twoTone(ary, nStd=2, preprocess='norm', invert=False, periods=1):
   assert preprocess in 'norm clip none'.split()
   if preprocess == 'norm':
       ary   = norm(ary, nStd)
   elif preprocess == 'clip':
       ary   = clip(ary)

   R, C      = ary.shape

   colorized = np.zeros((R, C, 3))
   if periods != 1:
      ary = np.abs(signal.sawtooth(periods*3.14159*ary))
   if invert:
      colorized[:, :, 0] = ary
      colorized[:, :, 1] = 1-ary
   else:
      colorized[:, :, 0] = 1-ary
      colorized[:, :, 1] = ary

   colorized *= (ary != 0)[:, :, None]

   return colorized
