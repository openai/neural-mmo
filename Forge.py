'''Main file for /projekt demo

The demo is simply a copy of my own research code.
It is not a simplest-possible working example, nor would
one be helpful -- the IO is a bit nontrivial in this setting,
and the model presented should give you a starting point for
dealing with it.'''

from pdb import set_trace as T
import argparse

from experiments import Experiment, Config
from forge.blade import lib

from forge.trinity import Trinity
from forge.ethyr.torch import Model

from projekt import Pantheon, God, Sword
from projekt.ann import ANN

from forge.trinity.ascend import Log

def parseArgs():
   '''Processes command line arguments'''
   parser = argparse.ArgumentParser('Projekt Godsword')
   parser.add_argument('--ray', type=str, default='default', 
         help='Ray mode (local/default/remote)')
   parser.add_argument('--render', action='store_true', default=False, 
         help='Render env')
   return parser.parse_args()

def render(trin, config, args):
   """Runs the environment in render mode

   Connect to localhost:8080 to view the client.

   Args:
      trin   : A Trinity object as shown in __main__
      config : A Config object as shown in __main__

   Notes:
      Blocks execution. This is an unavoidable side
      effect of running a persistent server with
      a fixed tick rate
   """

   from forge.embyr.twistedserver import Application

   #Prevent accidentally overwriting the trained model
   config.LOAD = True
   config.TEST = True
   config.BEST = True

   #Init infra in local mode
   args.ray = 'local'
   lib.ray.init(config, args.ray)

   #Instantiate environment
   god   = trin.god.remote(trin, config, idx=0)

   #Load model
   model = Model(ANN, config).load(None, config.BEST).weights
   
   #Pass the tick thunk to a twisted WebSocket server
   env = god.getEnv.remote()
   god.tick.remote(model)

   #Decision making is currently flawed.
   #The number of stims/actions keeps going up,
   #and entities are not being marked dead.
   Application(env, god.tick.remote)

if __name__ == '__main__':
   #Set up experiment configuration
   config = Experiment('pop', Config).init()
   args   = parseArgs()

   #Blocking call: switches execution to a
   #Web Socket Server module
   if args.render:
      render(trinity, config, args)


   #Trinity specifies Cluster-Server-Core infra modules
   trinity = Trinity(Pantheon, God, Sword).init(config, args)

   while True:
      log = trinity.step()
      print(log)
