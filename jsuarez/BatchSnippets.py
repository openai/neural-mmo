#         ann, rets = self.anns[0], []
#         for entID, ent, stim in ents:
#            annID = hash(entID) % self.nANN
#            unpacked = unpackStim(ent, stim)
#            self.anns[annID].recv(unpacked, ent, stim, entID)
#
#         #Ret order matters
#         for logits, val, ent, stim, atn, entID in ann.send():
#            action, args = self.actionArgs(stim, ent, atn.item())
#            rets.append((action, args, float(val)))
#            if ent.alive and not self.args.test:
#               self.collectStep(entID, (logits, val, atn))
#         return rets

class CosineNet(nn.Module):
   def __init__(self, xdim, h, ydim):
      super().__init__()
      self.feats = FeatNet(xdim, h, ydim)
      self.fc1 = torch.nn.Linear(h, h)
      self.ent1 = torch.nn.Linear(5, h)

   def forward(self, stim, conv, flat, ents, ent, actions):
      x = self.feats(conv, flat, ents)
      x = self.fc1(x)
      arguments = actions.args(stim, ent)
      ents = torch.tensor(np.array([e.stim for
          e in arguments])).float()
      args = self.ent1(ents) #center this in preprocess

      arg, argIdx = CosineClassifier(x, args)
      argument = [arguments[int(argIdx)]]
      return actions, argument, (arg, argIdx)

def CosineClassifier(x, a):
   ret = torch.sum(x*a, dim=1).view(1, -1)
   return ret, classify(ret)

class AtnNet(nn.Module):
   def __init__(self, xdim, h, ydim):
      super().__init__()
      self.feats = FeatNet(xdim, h, ydim)
      self.atn1 = torch.nn.Linear(h, 2)

   def forward(self, conv, flat, ent, flatEnts, actions):
      x = self.feats(conv, flat, flatEnts)
      atn = self.atn1(x)
      atnIdx  = classify(atn)
      return x, atn, atnIdx

class ActionEmbed(nn.Module):
   def __init__(self, nEmbed, dim):
      super().__init__()
      self.embed = torch.nn.Embedding(nEmbed, dim)
      self.atnIdx = {}

   def forward(self, actions):
      idxs = []
      for a in actions:
          if a not in self.atnIdx:
              self.atnIdx[a] = len(self.atnIdx)
          idxs.append(self.atnIdx[a])
      idxs = torch.tensor(idxs)
      atns = self.embed(idxs)
      return atns

def vDiffs(v):
   pad = v[0] * 0
   diffs = [vNew - vOld for vNew, vOld in zip(v[1:], v[:-1])]
   vRet = diffs + [pad]
   return vRet

def embedArgsLists(argsLists):
   args = [embedArgs(args) for args in argsLists]
   return np.stack(args)

def embedArgs(args):
   args = [embedArg(arg) for arg in args]
   return np.concatenate(args)

def embedArg(arg):
   arg = Arg(arg)
   arg = oneHot(arg.val - arg.min, arg.n)
   return arg

def matOneHot(mat, dim):
   r, c = mat.shape
   x = np.zeros((r, c, dim))
   for i in range(r):
      for j in range(c):
         x[i, j, mat[i,j]] = 1
   return x

#Old unzip. Don't use. Soft breaks PG
def unzipRollouts(rollouts):
   atnArgList, atnArgIdxList, valList, rewList = [], [], [], []
   for atnArgs, val, rew in rollouts:
      for atnArg, idx in atnArgs:
         atnArgList.append(atnArg)
         atnArgIdxList.append(idx)
         valList.append(val)
         rewList.append(rew)
   atnArgs    = atnArgList
   atnArgsIdx = torch.stack(atnArgIdxList)
   vals       = torch.stack(valList).view(-1, 1)
   rews       = torch.tensor(rewList).view(-1, 1).float()
   return atnArgs, atnArgsIdx, vals, rews

def l1Range(ent, sz, me, rng):
   R, C = sz
   rs, cs = me.pos
   rt = max(0, rs-rng)
   rb = min(R, rs+rng+1)
   cl = max(0, cs-rng)
   cr = min(C, cs+rng+1)
   ret = []
   for r in range(rt, rb):
      for c in range(cl, cr):
         if me in ent[r, c].ents:
             continue
         if len(ent[r, c].ents) > 0:
            ret += ent[r, c].ents
   return ret
