'''Main file for neural-mmo/projekt demo

In lieu of a simplest possible working example, I
have chosen to provide a fully featured copy of my
own research code. Neural MMO is persistent with 
large and variably sized agent populations -- 
features not present in smaller scale environments.
This causes differences in the training loop setup
that are fundamentally necessary in order to maintain
computational efficiency. As such, it is most useful
to begin by considering this full example.

I have done my best to structure the demo code
heirarchically. Reading only pantheon, god, and sword 
in /projeckt will give you a basic sense of the training
loop, infrastructure, and IO modules for handling input 
and output spaces. From there, you can either use the 
prebuilt IO networks in PyTorch to start training your 
own models immediately or dive deeper into the 
infrastructure and IO code.'''

#My favorite debugging macro
from pdb import set_trace as T 

import argparse

from forge.blade import lib
from forge.trinity import Trinity
from forge.ethyr.torch import Model

from experiments import Experiment, Config
from projekt import Pantheon, God, Sword
from projekt.ann import ANN

def parseArgs():
   '''Processes command line arguments'''
   parser = argparse.ArgumentParser('Projekt Godsword')
   parser.add_argument('--ray', type=str, default='default', 
         help='Ray mode (local/default/remote)')
   parser.add_argument('--render', action='store_true', default=False, 
         help='Render env')
   return parser.parse_args()

def render(trinity, config, args):
   """Runs the environment with rendering enabled. To pull
   up the Unity client, run ./client.sh in another shell.

   Args:
      trinity : A Trinity object as shown in __main__
      config  : A Config object as shown in __main__
      args    : Command line arguments from argparse

   Notes:
      Rendering launches a WebSocket server with a fixed tick
      rate. This is a blocking call; the server will handle 
      environment execution using the provided tick function.
   """

   #Prevent accidentally overwriting the trained model
   config.LOAD = True
   config.TEST = True
   config.BEST = True

   #Init infra in local mode
   lib.ray.init(config, 'local')

   #Instantiate environment and load the model,
   #Pass the tick thunk to a twisted WebSocket server
   god   = trin.god.remote(trin, config, idx=0)
   model = Model(ANN, config).load(None, config.BEST).weights
   env   = god.getEnv.remote()
   god.tick.remote(model)

   #Start a websocket server for rendering. This requires
   #forge/embyr, which is automatically downloaded from
   #jsuarez5341/neural-mmo-client in scripts/setup.sh
   from forge.embyr.twistedserver import Application
   Application(env, god.tick.remote)

if __name__ == '__main__':
   #Experiment + command line args specify configuration
   #Trinity specifies Cluster-Server-Core infra modules
   config  = Experiment('pop', Config).init()
   trinity = Trinity(Pantheon, God, Sword)
   args    = parseArgs()

   #Blocking call: switches execution to a
   #Web Socket Server module
   if args.render:
      render(trinity, config, args)

   #Train until AGI emerges
   trinity.init(config, args)
   while True:
      log = trinity.step()
      print(log)
