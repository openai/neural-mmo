from pdb import set_trace as T
import numpy as np

def unpack(vals, lens, dim=-1):
   '''Internal nD tensor pack utility'''
   if dim < 0:
      dim = len(vals.shape) + dim

   ret = []
   prefix = [slice(0, l) for l in vals.shape[1:dim]]
   suffix = [slice(0, l) for l in vals.shape[dim+1:]]
   for idx, l in enumerate(lens):
      middle = [slice(0, l)]
      inds = tuple([idx] + prefix + middle + suffix)
      e = vals[inds]
      ret.append(e)
   return ret

def pack(val):
   '''Internal nD tensor unpack utility'''
   shapes = np.array([e.shape for e in val])
   shape  = np.max(shapes, 0)

   #Must pad with zero (not -1) to satisfy embedding constraints
   dtype = val[0].dtype
   seq_tensor = np.zeros((len(val), *shape), dtype=dtype)
   seq_lens   = np.array([len(e) for e in val])

   for idx, tensor in enumerate(val):
      idx = tuple([idx] + [slice(0, i) for i in tensor.shape])
      seq_tensor[idx] = tensor
      
   return seq_tensor, seq_lens


