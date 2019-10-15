from pdb import set_trace as T
import sys, os
from experiments import Config

with open('scripts/hosts.txt') as f:
   hosts = f.read().splitlines()
   server = hosts[3]
   client = hosts[6:]
   client = ' '.join(client)

PORT       = '6379'
PREFIX_CMD = 'ray stop; ray start'


#ray stop; ray start --head --redis-port=6379 --num-cpus=0
SERVER_CMD = ' '.join([
      PREFIX_CMD,
      '--head', 
      '--redis-port='+PORT,
      '--num-cpus=0'
      ])

#parallel-ssh -h hosts.txt -P -i "ray stop; ray start --block --address=vision33:6379 --num-cpus=12"
CLIENT_CMD = ' '.join([
      'parallel-ssh --host', client, '-P -i',
      '"' + PREFIX_CMD, 
      '--block',
      '--address='+server+':'+PORT,
      '--num-cpus='+str(Config.NGOD) 
      + '"',
      ])

print(SERVER_CMD)
os.system(SERVER_CMD)

print(CLIENT_CMD)
os.system(CLIENT_CMD)
