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
from projekt.timed import Summary
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

def render(trin, config):
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

   #Note: this is a small hack to reuse training code
   #at test time in order to avoid rewriting the
   #lengthy inference loo
   god   = trin.god.remote(trin, config, idx=0)
   model = Model(ANN, config)

   #Load model
   model.load(None, config.BEST)
   packet = model.weights
   
   #Pass the tick thunk to a twisted WebSocket server
   env = god.getEnv.remote()
   god.tick.remote(packet)
   Application(env, god.tick.remote)

if __name__ == '__main__':
   #Set up experiment configuration
   #ray infra, and command line args
   config = Experiment('env', Config).init(
      NPOP=1,
      NENT=128,
   )

   args = parseArgs()

   if args.render:
      args.ray = 'local'
   lib.ray.init(args.ray)

   #Create a Trinity object specifying
   #Cluster, Server, and Core level execution
   trinity = Trinity(Pantheon, God, Sword)

   if args.render:
      render(trinity, config)

   trinity.init(config)

   #Run and print logs
   while True:
      log = trinity.step()
      log = Log.summary([log, trinity.discipleLogs()])
      summary = Summary(log)
      print(str(summary))
