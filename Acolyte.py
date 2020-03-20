from pdb import set_trace as T
from collections import defaultdict

import numpy as np
import ray

from forge.blade import core
from forge.blade.lib.ray import init
from forge.blade.io.action import static as action

from forge.ethyr.torch import param

import torch
from torch import nn
from torch.distributions import Categorical

def oneHot(i, n):
   vec = [0 for _ in range(n)]
   vec[i] = 1
   return vec

class Config(core.Config):
   NENT    = 256
   NPOP    = 8

   NREALM  = 256
   NWORKER = 4

   EMBED   = 32
   HIDDEN  = 64
   WINDOW  = 4

   OPTIM_STEPS = 128

class Policy(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.embed  = nn.Linear(492, config.EMBED)
      self.hidden = nn.Linear(config.EMBED, config.HIDDEN)
      #self.hidden  = nn.LSTM(config.EMBED, config.HIDDEN)
      self.action = nn.Linear(config.HIDDEN, 4)

   def forward(self, x):
      x = torch.tensor(x).float()
      x = self.embed(x)
      x = self.hidden(x)
      x = self.action(x)
      return x

class Realm(core.Realm):
   '''Example environment overrides'''
   def step(self, decisions):
      ents = []
      stims, rewards, dones, _ = super().step(decisions)
      for idx, stim in enumerate(stims):
         env, ent = stim
         conf = self.config
         n = conf.STIM
         s = []

         for i in range(n-conf.WINDOW, n+conf.WINDOW+1):
            for j in range(n-conf.WINDOW, n+conf.WINDOW+1):
               tile = env[i, j]
               s += oneHot(tile.mat.index, 6)

         s.append(ent.resources.health.val)
         s.append(ent.resources.health.max)
         s.append(ent.resources.food.val)
         s.append(ent.resources.food.max)
         s.append(ent.resources.water.val)
         s.append(ent.resources.water.max)

         stims[idx] = s
         ents.append(ent)

      return ents, stims, rewards, dones

class Optim:
   def __init__(self, config):
      self.config = config
      self.net    = Policy(config).eval()

   def run(self):
      config = self.config
      while True:
         params = param.getParameters(self.net)
         params = np.array(params)
         returns = []
         for _ in range(config.NWORKER):
            worker = Worker.remote(config)
            ret    = worker.run.remote(params)
            returns.append(ret)
         data = ray.get(returns)
         returns = []
         for dat in data:
            returns += dat

         grad = np.zeros_like(params)
         rewards = []
         for entID, reward in returns:
            rewards.append(reward)
            np.random.seed(entID)
            noise = reward * np.random.randn(len(params))
            grad += noise
         
         print('Reward: {:.2f}'.format(np.mean(rewards)))
         grad = grad / len(data)
         params += 0.01 * grad
         param.setParameters(self.net, params)
            

@ray.remote
class Worker:
   def __init__(self, config):
      idx = np.random.randint(config.NREALM)
      self.net = Policy(config).eval()
      self.env = Realm(config, idx)

   def run(self, params):
      ents, obs, rewards, dones = self.env.reset()
      returns = []
      for _ in range(config.OPTIM_STEPS):
         #Collect rollouts
         for ent in dones:
            returns.append((ent.entID, ent.history.timeAlive.val-1))

         actions = {}
         for ent, ob in zip(ents, obs):
            #Perturbed rollout. Should salt the entID per realm
            np.random.seed(ent.entID)
            noise = np.random.randn(len(params))
            param.setParameters(self.net,  params + 0.01*noise)
            atn = self.net(ob)

            #Postprocess actions
            distribution = Categorical(logits=atn)
            atn = distribution.sample()
            arg = action.Direction.edges[int(atn)]
            actions[ent.entID] = {action.Move: [arg]}

         #Step environment
         ents, obs, rewards, dones = self.env.step(actions)

      return returns

if __name__ == '__main__':
   #init(None, mode='local')
   ray.init()
   config = Config()
   Optim(config).run()

