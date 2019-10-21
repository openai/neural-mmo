from pdb import set_trace as T
import sys, os
from experiments import Config

with open('scripts/private_hosts.txt') as f:
   hosts = f.read().splitlines()
   server = hosts[3]
   client = hosts[6:]
   client = [c for c in client if c[0] != '#']
   client = ' '.join(client)

PORT       = '6379'
PREFIX_CMD = 'ray stop; ray start'


#ray stop; ray start --head --redis-port=6379 --num-cpus=0
SERVER_CMD = ' '.join([
      'ray stop; MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ray start',
      '--head', 
      '--redis-port='+PORT,
      '--num-cpus='+str(Config.NCORE)
      ])

#parallel-ssh -h hosts.txt -P -i "ray stop; ray start --block --address=vision33:6379 --num-cpus=12"
#      '"' + 'ray stop; bash ~/longjob -u jsuarez -k ~/jsuarez.keytab MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ray start',
CLIENT_CMD = ' '.join([
      'parallel-ssh --host', client, '-P -i',
      '"' + 'ray stop; MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ray start',
      '--block',
      '--address='+server+':'+PORT,
      '--num-cpus='+str(Config.NCORE) 
      + '"',
      ])

print(SERVER_CMD)
os.system(SERVER_CMD)

print(CLIENT_CMD)
os.system(CLIENT_CMD)
