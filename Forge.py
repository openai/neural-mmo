'''Main file for /projekt demo'''
from pdb import set_trace as T
import argparse

from experiments import Experiment, Config
from forge.blade import lib

from forge.trinity import smith, Trinity
from forge.trinity.timed import TimeLog

from projekt import Pantheon, God, Sword

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
   sword = trin.sword.remote(trin, config, args, idx=0)
   env = sword.getEnv.remote()
   Application(env, sword.tick.remote)

if __name__ == '__main__':
   #Set up experiment configuration
   #ray infra, and command line args
   config = Experiment('demo', Config).init(
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
      render(trinity, config, args)

   trinity.init(config, args)

   #Run and print logs
   while True:
      time = trinity.step()
      logs = trinity.logs()
      logs = TimeLog.log(logs)
