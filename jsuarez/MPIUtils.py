#Author: Joseph Suarez

from pdb import set_trace as T
import sys, builtins
import numpy as np

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

MASTER  = 0
SILENT  = 1
ALL     = 2

class LoadBalancer:
   def __init__(self, cores):
      self.nCores = len(cores)
      self.cores  = cores
      self.loads  = dict((core, 0) for core in cores)

   def assignWorker(self):
      #core = 1
      #self.loads[core] += 1
      #return np.random.choice(self.cores)
      #return min([len(e) for e in load.values()])
      core = min(self.loads, key=self.loads.get)
      self.loads[core] += 1
      return core

   def deleteWorker(self, core):
       self.loads[core] -= 1

def print(verbose, *args):
   if verbose == ALL or (verbose == MASTER and isMaster()):
      builtins.print(*args)
      sys.stdout.flush()

def send(data, dst, seq=None, usePar=False):
   if not usePar:
      seq.inbox = data
      return
   comm.send(data, dst)

def recv(src, seq=None, usePar=False):
   if not usePar:
      return seq.inbox
   return comm.recv(source=src)

#Returns a req
def isend(data, dst, tag):
   return comm.isend(data, dest=dst, tag=tag)

#Returns a req
def irecv(src, tag):
   return comm.irecv(source=src, tag=tag)

def gather(dst):
   return comm.gather(root=dst)

def assignWorker(clients):
   return np.random.choice(clients)

def distributeFunc(f):
   if isMaster():
      x = f()
   else:
      x = None
   return distribute(x)

def npValMean(val):
   meanVal = np.zeros_like(val)
   comm.Allreduce(val, meanVal, op=MPI.SUM)
   return meanVal / comm.Get_size()

def distribute(x):
   return comm.bcast(x, root=MASTER)

def isMaster():
   return comm.Get_rank() == MASTER

def core():
   return comm.Get_rank()
