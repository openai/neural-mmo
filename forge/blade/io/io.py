from pdb import set_trace as T
from collections import defaultdict
import time

from forge.blade.io import stimulus, action
from forge.blade.io.serial import Serial
from forge.blade.io import utils

### Begin Internal IO Packet Objects ###
class Output:
   def __init__(self, key, atn, atnIdx, value):
      self.key    = key
      self.out    = (key, atnIdx, atn)

      self.action = {key: atnIdx}
      self.value  = value

class ObsData:
   def __init__(self):
      self.entities = defaultdict(EntityData)
      self.names = defaultdict(list)
      self.n = 0

class EntityData:
   def __init__(self):
      self.attributes = defaultdict(list)

class AtnData:
   def __init__(self):
      self.actions = defaultdict(ArgsData)
      self.names = []

class ArgsData:
   def __init__(self):
      self.arguments = defaultdict(list)
### End Internal IO Packet Objects ###

class IOPacket:
   '''High level I/O class for abstracting game state and action selection'''
   def __init__(self):
      self.obs    = ObsData()
      self.atn    = AtnData()
      self.lookup  = Lookup()

      self.keys    = []
      self.rewards = []
      self.dones   = []

   def actions(self, serialize=True):
      for atn in action.Static.arguments:
         serial = atn
         if serialize:
            serial = Serial.key(serial)
         self.lookup.add(serial, orig=atn)

   def key(self, env, ent, reward, config):
      annID, entID = ent.annID, ent.entID
      key = (annID, entID)
      self.keys.append(key)
      self.rewards.append(reward)

   def pack(self):
      for atn, action in self.atn.actions.items():
         for arg, argument in action.arguments.items():
            tensor, lens = utils.pack(argument)
            self.atn.actions[atn].arguments[arg] = tuple([tensor, lens])

class IO:
   '''Core IO API for pre/postprocessing environment data'''
   def inputs(obs, rewards, dones, config, 
            clientHash=None, serialize=True):
      '''Core Input library for preprocessing, batching, and serializing
      environment observations and data for use in training and inference.

      Args:
         obs: A list of observations as specified by
            :func:`~forge.blade.core.api.Realm.step`
         rewards: A list of rewards as specified by
            :func:`~forge.blade.core.api.Realm.step`
         dones: A list of "done" (agent/episode termination signals) as
            specified by :func:`~forge.blade.core.api.Realm.step`
         config: An environment :class:`~forge.blade.core.config.Config`
            object used to specify game parameters
         clientHash: An integer-integer hash function operating over entity
            IDs for sharded training (e.g. specifies which shard each agent
            belongs to when using multiple remote clients per server). If not
            specified, no sharding is performed.
         serialize:
            Boolean specifying whether to serialize environment observations.
            This should only be set to False for debugging purposes.

      Returns:
         (defaultdict<IOPacket>, int):

         clientData:
            IOPacket objects populated with structued obs/rewards/dones data
            required for efficient inference and training. Keys correspond to
            the indices of remote shards/clients. If clientHash is None (e.g.
            sharding is disabled), this return value will instead be a single
            IOPacket object instead of a defaultdict of them. 

         nUpdates:
            The total number of agent observations returned by obs.

      Notes:
         IOPacket objects are fairly complex and abstract preprocessing,
         batching, and serialization. Type checking the contents of
         obs/rewards/dones is difficult and slow. You should have no issues
         as long as you obey the function spec but may get internal errors
         if you try to hack the IOPacket structure.
      '''
      inputs = defaultdict(IOPacket)

      #No sharding
      default = clientHash is None
      if clientHash is None:
         clientHash=lambda x: 0

      for done in dones:
         idx = clientHash(done[1])
         inputs[idx].dones.append(done)

      ### Process inputs
      n = 0
      for ob, reward in zip(obs, rewards):
         env, ent = ob
         idx = clientHash(ent.entID)
         inputs[idx].key(env, ent, reward, config)
         stimulus.Dynamic.process(config, inputs[idx], env, ent, serialize)
         inputs[idx].obs.n += 1
         n += 1
      
      start = time.time()
      #Index actions
      for idx, inp in inputs.items():
         inputs[idx].actions()

      ### Process outputs
      for ob, reward in zip(obs, rewards):
         env, ent = ob
         idx = clientHash(ent.entID)
         action.Dynamic.process(inputs[idx], env, ent, config, serialize)

      #Pack actions
      for idx, inp in inputs.items():
         inputs[idx].pack()

      #No sharding
      if default:
         inputs = inputs[0]

      return inputs, n

   def outputs(obs, atnDict=None):
      '''Core Output library for postprocessing flat agent decisions into
      structured action argument lists

      Args:
         obs: A list of observations as specified by
            :func:`~forge.blade.core.api.Realm.step`
         atnDict: Accumulated using the returned atnDict of this function across
            multiple calls (format below). None upon first call.

      Returns:
         defaultdict<defaultdict<list>>:

         atnDict:
            Populated dictionary of actions of format::

               {
                  ent_1: {
                     action_1: [arg_1, arg_2],
                     action_2: [...],
                     ...
                  },
                  ent_2: {
                     ...
                  },
                  ...
               }
      '''
      #Initialize output dictionary
      if atnDict is None:
         atnDict = defaultdict(lambda: defaultdict(list))

      #Reverse format lookup over actions
      names = list(obs.obs.names.keys())
      for atn, action in obs.atn.actions.items():
         for arg, atnsIdx in action.arguments.items():
            for idx, a in enumerate(atnsIdx):
               _, entID, _ = names[idx]
               a = obs.lookup.reverse(a)
               atnDict[entID][atn].append(a)

      return atnDict 

class Lookup:
   '''Lookup utility for indexing 
   (name, data) pairs'''
   def __init__(self):
      self.data = {}
      self.back = {}
      self.max = 0

   def add(self, name, idx=None, orig=None):
      '''Add entries to the table'''
      if idx is None:
         idx = self.max

      assert name not in self.data
      assert idx not in self.back
      self.data[name] = idx
      if orig is not None:
         self.back[idx] = orig
      self.max += 1

      return idx

   def __contains__(self, key):
      return key in self.data

   def reverse(self, idx):
      return self.back[idx]
