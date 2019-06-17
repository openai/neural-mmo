from pdb import set_trace as T
import numpy as np

def deprecate_pack(val):
   seq_lens   = list(map(len, val))
   seq_tensor = np.zeros((len(val), max(seq_lens)))
   for idx, (seq, seqlen) in enumerate(zip(val, seq_lens)):
      seq_tensor[idx, :seqlen] = np.array(seq)

   #Todo: reintroduce sort
   #seq_lens, perm_idx = seq_lens.sort(0, descending=True)
   #seq_tensor = seq_tensor[perm_idx]

   return seq_tensor, seq_lens

#Be sure to unsort these
def unpack(vals, lens, dim=-1):
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


