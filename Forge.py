"""
Forge
====================================
Initial docstring test
"""

#Main file. Hooks into high level world/render updates
from pdb import set_trace as T
import argparse

import experiments
from forge.trinity import smith, Trinity, Pantheon, God, Sword
from forge.trinity.timed import TimeLog
from forge.blade import lib

def parseArgs():
   parser = argparse.ArgumentParser('Projekt Godsword')
   parser.add_argument('--nRealm', type=int, default='1', 
         help='Number of environments (1 per core)')
   parser.add_argument('--api', type=str, default='native', 
         help='API to use (native/vecenv)')
   parser.add_argument('--ray', type=str, default='default', 
         help='Ray mode (local/default/remote)')
   parser.add_argument('--render', action='store_true', default=False, 
         help='Render env')
   return parser.parse_args()

def render(trin, config, args):
   """
   Runs the environment in render mode

   Parameters
   ---------
   trin 
      A Trinity object to create the envionment
   config
      A Configuration to use

   """
   from forge.embyr.twistedserver import Application
   sword = trin.sword.remote(trin, config, args, idx=0)
   env = sword.getEnv.remote()
   Application(env, sword._step.remote)

if __name__ == '__main__':
   args = parseArgs()
   assert args.api in ('native', 'vecenv')
   config = experiments.exps['nxt-auto-treechaos128']

   lib.ray.init(args.ray)
   trin = Trinity(Pantheon, God, Sword)

   #Rendering by necessity snags control flow
   #This will automatically set local mode with 1 core
   if args.render:
      render(trin, config, args)

   trin.init(config, args)

   while True:
      time = trin.step()
      logs = trin.logs()
      logs = TimeLog.log(logs)
