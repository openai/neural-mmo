from collections import defaultdict

from tqdm import tqdm
import numpy as np
import gym
import wandb
import trueskill

import torch
from torch import nn
from torch.nn.utils import rnn

from ray import rllib
from pdb import set_trace as TT

import ray.rllib.agents.ppo.ppo as ppo
import ray.rllib.agents.ppo.appo as appo
import ray.rllib.agents.impala.impala as impala
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

from neural_mmo.forge.blade.io.action.static import Action, Fixed
from neural_mmo.forge.blade.io.stimulus.static import Stimulus
from neural_mmo.forge.blade.lib import overlay
from neural_mmo.forge.blade.systems import ai

from neural_mmo.forge.ethyr.torch import policy
from neural_mmo.forge.ethyr.torch.policy import attention

from neural_mmo.forge.trinity import Env
from neural_mmo.forge.trinity.dataframe import DataType
from neural_mmo.forge.trinity.overlay import Overlay, OverlayRegistry


###############################################################################
### Pytorch model + IO. The pip package contains some submodules
class Input(nn.Module):
   def __init__(self, config, embeddings, attributes):
      '''Network responsible for processing observations

      Args:
         config     : A configuration object
         embeddings : An attribute embedding module
         attributes : An attribute attention module
      '''
      super().__init__()

      self.embeddings = nn.ModuleDict()
      self.attributes = nn.ModuleDict()

      for _, entity in Stimulus:
         continuous = len([e for e in entity if e[1].CONTINUOUS])
         discrete   = len([e for e in entity if e[1].DISCRETE])
         self.attributes[entity.__name__] = nn.Linear(
               (continuous+discrete)*config.HIDDEN, config.HIDDEN)
         self.embeddings[entity.__name__] = embeddings(
               continuous=continuous, discrete=4096, config=config)

      #Hackey obs scaling
      self.tileWeight = torch.Tensor([1.0, 0.0, 0.02, 0.02])
      self.entWeight  = torch.Tensor([1.0, 0.0, 0.0, 0.05, 0.00, 0.02, 0.02, 0.1, 0.01, 0.1, 0.1, 0.1, 0.3])

   def forward(self, inp):
      '''Produces tensor representations from an IO object

      Args:                                                                   
         inp: An IO object specifying observations                      
         
      Returns:
         entityLookup: A fixed size representation of each entity
      ''' 
      #Pack entities of each attribute set
      entityLookup = {}

      device                       = inp['Tile']['Continuous'].device
      inp['Tile']['Continuous']   *= self.tileWeight.to(device)
      inp['Entity']['Continuous'] *= self.entWeight.to(device)
 
      entityLookup['N'] = inp['Entity'].pop('N')
      for name, entities in inp.items():
         #Construct: Batch, ents, nattrs, hidden
         embeddings = self.embeddings[name](entities)
         B, N, _, _ = embeddings.shape
         embeddings = embeddings.view(B, N, -1)

         #Construct: Batch, ents, hidden
         entityLookup[name] = self.attributes[name](embeddings)

      return entityLookup

class Output(nn.Module):
   def __init__(self, config):
      '''Network responsible for selecting actions

      Args:
         config: A Config object
      '''
      super().__init__()
      self.config = config
      self.h = config.HIDDEN

      self.net = DiscreteAction(self.config, self.h, self.h)
      self.arg = nn.Embedding(Action.n, self.h)

   def names(self, nameMap, args):
      '''Lookup argument indices from name mapping'''
      return np.array([nameMap.get(e) for e in args])

   def forward(self, obs, lookup):
      '''Populates an IO object with actions in-place                         
                                                                              
      Args:                                                                   
         obs     : An IO object specifying observations
         lookup  : A fixed size representation of each entity
      ''' 
      rets = defaultdict(dict)
      for atn in Action.edges:
         for arg in atn.edges:
            lens  = None
            if arg.argType == Fixed:
               batch = obs.shape[0]
               idxs  = [e.idx for e in arg.edges]
               cands = self.arg.weight[idxs]
               cands = cands.repeat(batch, 1, 1)
            else:
               cands = lookup['Entity']
               lens  = lookup['N']

            logits = self.net(obs, cands, lens)
            rets[atn][arg] = logits

      return rets
      
class DiscreteAction(nn.Module):
   '''Head for making a discrete selection from
   a variable number of candidate actions'''
   def __init__(self, config, xdim, h):
      super().__init__()
      self.net = attention.DotReluBlock(h)

   def forward(self, stim, args, lens):
      x = self.net(stim, args)

      if lens is not None:
         mask = torch.arange(x.shape[-1]).to(x.device).expand_as(x)
         x[mask >= lens] = 0

      return x

class Base(nn.Module):
   def __init__(self, config):
      '''Base class for baseline policies

      Args:
         config: A Configuration object
      '''
      super().__init__()
      self.embed  = config.EMBED
      self.config = config

      self.output = Output(config)
      self.input  = Input(config,
            embeddings=policy.MixedDTypeInput,
            attributes=policy.SelfAttention)

      self.valueF = nn.Linear(config.HIDDEN, 1)

   def hidden(self, obs, state=None, lens=None):
      '''Abstract method for hidden state processing, recurrent or otherwise,
      applied between the input and output modules

      Args:
         obs: An observation dictionary, provided by forward()
         state: The previous hidden state, only provided for recurrent nets
         lens: Trajectory segment lengths used to unflatten batched obs
      '''
      raise NotImplementedError('Implement this method in a subclass')

   def forward(self, obs, state=None, lens=None):
      '''Applies builtin IO and value function with user-defined hidden
      state subnetwork processing. Arguments are supplied by RLlib
      '''
      entityLookup  = self.input(obs)
      hidden, state = self.hidden(entityLookup, state, lens)
      self.value    = self.valueF(hidden).squeeze(1)
      actions       = self.output(hidden, entityLookup)
      return actions, state

class Encoder(Base):
   def __init__(self, config):
      '''Simple baseline model with flat subnetworks'''
      super().__init__(config)
      h = config.HIDDEN

      self.ent    = nn.Linear(2*h, h)
      self.conv   = nn.Conv2d(h, h, 3)
      self.pool   = nn.MaxPool2d(2)
      self.fc     = nn.Linear(h*6*6, h)

      self.proj   = nn.Linear(2*h, h)
      self.attend = policy.SelfAttention(self.embed, h)

   def hidden(self, obs, state=None, lens=None):
      #Attentional agent embedding
      agentEmb  = obs['Entity']
      selfEmb   = agentEmb[:, 0:1].expand_as(agentEmb)
      agents    = torch.cat((selfEmb, agentEmb), dim=-1)
      agents    = self.ent(agents)
      agents, _ = self.attend(agents)
      #agents = self.ent(selfEmb)

      #Convolutional tile embedding
      tiles     = obs['Tile']
      self.attn = torch.norm(tiles, p=2, dim=-1)

      w      = self.config.WINDOW
      batch  = tiles.size(0)
      hidden = tiles.size(2)
      #Dims correct?
      tiles  = tiles.reshape(batch, w, w, hidden).permute(0, 3, 1, 2)
      tiles  = self.conv(tiles)
      tiles  = self.pool(tiles)
      tiles  = tiles.reshape(batch, -1)
      tiles  = self.fc(tiles)

      hidden = torch.cat((agents, tiles), dim=-1)
      hidden = self.proj(hidden)
      return hidden, state

class Recurrent(Encoder):
   def __init__(self, config):
      '''Recurrent baseline model'''
      super().__init__(config)
      self.lstm   = policy.BatchFirstLSTM(
            input_size=config.HIDDEN,
            hidden_size=config.HIDDEN)

   #Note: seemingly redundant transposes are required to convert between 
   #Pytorch (seq_len, batch, hidden) <-> RLlib (batch, seq_len, hidden)
   def hidden(self, obs, state, lens):
      #Attentional input preprocessor and batching
      lens = lens.cpu() if type(lens) == torch.Tensor else lens
      hidden, _ = super().hidden(obs)
      config    = self.config
      h, c      = state

      TB = hidden.size(0) #Padded batch of size (seq x batch)
      B  = len(lens)      #Sequence fragment time length
      T  = TB // B        #Trajectory batch size
      H  = config.HIDDEN  #Hidden state size

      #Pack (batch x seq, hidden) -> (batch, seq, hidden)
      hidden        = rnn.pack_padded_sequence(
                         input=hidden.view(B, T, H),
                         lengths=lens,
                         enforce_sorted=False,
                         batch_first=True)

      #Main recurrent network
      oldHidden = hidden
      hidden, state = self.lstm(hidden, state)
      newHidden = hidden

      #Unpack (batch, seq, hidden) -> (batch x seq, hidden)
      hidden, _     = rnn.pad_packed_sequence(
                         sequence=hidden,
                         batch_first=True,
                         total_length=T)

      return hidden.reshape(TB, H), state

###############################################################################
### RLlib Policy, Evaluator, Trainer
class RLlibPolicy(RecurrentNetwork, nn.Module):
   '''Wrapper class for using our baseline models with RLlib'''
   def __init__(self, *args, **kwargs):
      self.config = kwargs.pop('config')
      super().__init__(*args, **kwargs)
      nn.Module.__init__(self)

      self.space  = actionSpace(self.config).spaces
      self.model  = Recurrent(self.config)

   #Initial hidden state for RLlib Trainer
   def get_initial_state(self):
      return [self.model.valueF.weight.new(1, self.config.HIDDEN).zero_(),
              self.model.valueF.weight.new(1, self.config.HIDDEN).zero_()]

   def forward(self, input_dict, state, seq_lens):
      logitDict, state = self.model(input_dict['obs'], state, seq_lens)

      logits = []
      #Flatten structured logits for RLlib
      for atnKey, atn in sorted(self.space.items()):
         for argKey, arg in sorted(atn.spaces.items()):
            logits.append(logitDict[atnKey][argKey])

      return torch.cat(logits, dim=1), state

   def value_function(self):
      return self.model.value

   def attention(self):
      return self.model.attn


###############################################################################
### RLlib Wrappers: Env, Overlays
class RLlibEnv(Env, rllib.MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']
      super().__init__(self.config)

   def reward(self, ent):
      config      = self.config

      ACHIEVEMENT = config.REWARD_ACHIEVEMENT
      SCALE       = config.ACHIEVEMENT_SCALE
      COOPERATIVE = config.COOPERATIVE

      individual  = 0 if ent.entID in self.realm.players else -1
      team        = 0

      if ACHIEVEMENT:
         individual += SCALE*ent.achievements.update(self.realm, ent, dry=True)
      if COOPERATIVE:
         nDead = len([p for p in self.dead.values() if p.population == ent.pop])
         team  = -nDead / config.TEAM_SIZE
      if COOPERATIVE and ACHIEVEMENT:
         pre, post = [], []
         for p in self.realm.players.corporeal.values():
            if p.population == ent.pop:
               pre.append(p.achievements.score(aggregate=False))
               post.append(p.achievements.update(
                     self.realm, ent, aggregate=False, dry=True))
        
         pre   = np.array(pre).max(0)
         post  = np.array(post).max(0)
         team += SCALE*(post - pre).sum()

      ent.achievements.update(self.realm, ent)

      alpha  = config.TEAM_SPIRIT
      return alpha*team + (1.0-alpha)*individual

   def step(self, decisions, preprocess=None, omitDead=False):
      preprocess = {entID for entID in decisions}
      obs, rewards, dones, infos = super().step(decisions, preprocess, omitDead)

      config = self.config
      dones['__all__'] = False
      test = config.EVALUATE or config.RENDER
      
      if config.EVALUATE:
         horizon = config.EVALUATION_HORIZON
      else:
         horizon = config.TRAIN_HORIZON

      population  = len(self.realm.players) == 0
      hit_horizon = self.realm.tick >= horizon
      
      if not config.RENDER and (hit_horizon or population):
         dones['__all__'] = True

      return obs, rewards, dones, infos

def observationSpace(config):
   obs = FlexDict(defaultdict(FlexDict))
   for entity in sorted(Stimulus.values()):
      nRows       = entity.N(config)
      nContinuous = 0
      nDiscrete   = 0

      for _, attr in entity:
         if attr.DISCRETE:
            nDiscrete += 1
         if attr.CONTINUOUS:
            nContinuous += 1

      obs[entity.__name__]['Continuous'] = gym.spaces.Box(
            low=-2**20, high=2**20, shape=(nRows, nContinuous),
            dtype=DataType.CONTINUOUS)

      obs[entity.__name__]['Discrete']   = gym.spaces.Box(
            low=0, high=4096, shape=(nRows, nDiscrete),
            dtype=DataType.DISCRETE)

   obs['Entity']['N']   = gym.spaces.Box(
         low=0, high=config.N_AGENT_OBS, shape=(1,),
         dtype=DataType.DISCRETE)
   return obs

def actionSpace(config):
   atns = FlexDict(defaultdict(FlexDict))
   for atn in sorted(Action.edges):
      for arg in sorted(atn.edges):
         n              = arg.N(config)
         atns[atn][arg] = gym.spaces.Discrete(n)
   return atns

class RLlibOverlayRegistry(OverlayRegistry):
   '''Host class for RLlib Map overlays'''
   def __init__(self, realm):
      super().__init__(realm.config, realm)

      self.overlays['values']       = Values
      self.overlays['attention']    = Attention
      self.overlays['tileValues']   = TileValues
      self.overlays['entityValues'] = EntityValues

class RLlibOverlay(Overlay):
   '''RLlib Map overlay wrapper'''
   def __init__(self, config, realm, trainer, model):
      super().__init__(config, realm)
      self.trainer = trainer
      self.model   = model

class Attention(RLlibOverlay):
   def register(self, obs):
      '''Computes local attentional maps with respect to each agent'''
      tiles      = self.realm.realm.map.tiles
      players    = self.realm.realm.players

      attentions = defaultdict(list)
      for idx, playerID in enumerate(obs):
         if playerID not in players:
            continue
         player = players[playerID]
         r, c   = player.pos

         rad     = self.config.NSTIM
         obTiles = self.realm.realm.map.tiles[r-rad:r+rad+1, c-rad:c+rad+1].ravel()

         for tile, a in zip(obTiles, self.model.attention()[idx]):
            attentions[tile].append(float(a))

      sz    = self.config.TERRAIN_SIZE
      data  = np.zeros((sz, sz))
      for r, tList in enumerate(tiles):
         for c, tile in enumerate(tList):
            if tile not in attentions:
               continue
            data[r, c] = np.mean(attentions[tile])

      colorized = overlay.twoTone(data)
      self.realm.register(colorized)

class Values(RLlibOverlay):
   def update(self, obs):
      '''Computes a local value function by painting tiles as agents
      walk over them. This is fast and does not require additional
      network forward passes'''
      players = self.realm.realm.players
      for idx, playerID in enumerate(obs):
         if playerID not in players:
            continue
         r, c = players[playerID].base.pos
         self.values[r, c] = float(self.model.value_function()[idx])

   def register(self, obs):
      colorized = overlay.twoTone(self.values[:, :])
      self.realm.register(colorized)

def zeroOb(ob, key):
   for k in ob[key]:
      ob[key][k] *= 0

class GlobalValues(RLlibOverlay):
   '''Abstract base for global value functions'''
   def init(self, zeroKey):
      if self.trainer is None:
         return

      print('Computing value map...')
      model     = self.trainer.get_policy('policy_0').model
      obs, ents = self.realm.dense()
      values    = 0 * self.values

      #Compute actions to populate model value function
      BATCH_SIZE = 128
      batch = {}
      final = list(obs.keys())[-1]
      for agentID in tqdm(obs):
         ob             = obs[agentID]
         batch[agentID] = ob
         zeroOb(ob, zeroKey)
         if len(batch) == BATCH_SIZE or agentID == final:
            self.trainer.compute_actions(batch, state={}, policy_id='policy_0')
            for idx, agentID in enumerate(batch):
               r, c         = ents[agentID].base.pos
               values[r, c] = float(self.model.value_function()[idx])
            batch = {}

      print('Value map computed')
      self.colorized = overlay.twoTone(values)

   def register(self, obs):
      print('Computing Global Values. This requires one NN pass per tile')
      self.init()

      self.realm.register(self.colorized)

class TileValues(GlobalValues):
   def init(self, zeroKey='Entity'):
      '''Compute a global value function map excluding other agents. This
      requires a forward pass for every tile and will be slow on large maps'''
      super().init(zeroKey)

class EntityValues(GlobalValues):
   def init(self, zeroKey='Tile'):
      '''Compute a global value function map excluding tiles. This
      requires a forward pass for every tile and will be slow on large maps'''
      super().init(zeroKey)


class RLlibTrainer(ppo.PPOTrainer):
   def __init__(self, config, env=None, logger_creator=None):
      super().__init__(config, env, logger_creator)
      self.env_config = config['env_config']['config']

      #1/sqrt(2)=76% win chance within beta, 95% win chance vs 3*beta=100 SR
      trueskill.setup(mu=1000, sigma=2*100/3, beta=100/3, tau=2/3, draw_probability=0)

      self.ratings = [{agent.__name__: trueskill.Rating(mu=1000, sigma=2*100/3)}
            for agent in self.env_config.EVAL_AGENTS]

      self.reset_scripted()

   def reset_scripted(self):
      for rating_dict in self.ratings:
         for agent, rating in rating_dict.items():
            if agent == 'Combat':
               rating_dict[agent] = trueskill.Rating(mu=1500, sigma=1)

   def post_mean(self, stats):
      for key, vals in stats.items():
          if type(vals) == list:
              stats[key] = np.mean(vals)

   def train(self):
      stats = super().train()
      self.post_mean(stats['custom_metrics'])
      return stats

   def evaluate(self):
      stat_dict = super().evaluate()
      stats = stat_dict['evaluation']['custom_metrics']
 
      ranks = {agent.__name__: -1 for agent in self.env_config.EVAL_AGENTS}
      for key in list(stats.keys()):
         if key.startswith('Rank_'):
             stat = stats[key]
             del stats[key]
             agent = key[5:]
             ranks[agent] = stat

      ranks = list(ranks.values())
      nEnvs = len(ranks[0])
      
      #Once RLlib adds better custom metric support,
      #there should be a cleaner way to divide episodes into blocks
      for i in range(nEnvs): 
         env_ranks = [e[i] for e in ranks]
         self.ratings = trueskill.rate(self.ratings, env_ranks)
         self.reset_scripted()

      for rating in self.ratings:
         key  = 'SR_{}'.format(list(rating.keys())[0])
         val  = list(rating.values())[0]
         stats[key] = val.mu
     
      return stat_dict


###############################################################################
### Logging
class RLlibLogCallbacks(DefaultCallbacks):
   def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
      assert len(base_env.envs) == 1, 'One env per worker'
      env    = base_env.envs[0]

      logs = env.terminal()
      for key, vals in logs['Stats'].items():
         episode.custom_metrics[key] = np.mean(vals)

      if not env.config.EVALUATE:
         return 

      agents = defaultdict(list)

      stats      = logs['Stats']
      policy_ids = stats['PolicyID']
      scores     = stats['Achievement']

      invMap = {agent.policyID: agent for agent in env.config.AGENTS}

      for policyID, score in zip(policy_ids, scores):
         policy = invMap[policyID]
         agents[policy].append(score)
      
      for agent in agents:
         agents[agent] = np.mean(agents[agent])

      policies = list(agents.keys())
      scores   = list(agents.values())

      idxs     = np.argsort(-np.array(scores))

      for rank, idx in enumerate(idxs):
          key = 'Rank_{}'.format(policies[idx].__name__)
          episode.custom_metrics[key] = rank

