from sim.lib import MPIUtils

def packets():
   return [i for i in range(10000)]

class Client:
   def __init__(self, server):
      self.server = server

   def step(self):
      MPIUtils.print(MPIUtils.ALL, 'Client Recv')
      packets = MPIUtils.recv(self.server, usePar=True)
      MPIUtils.print(MPIUtils.ALL, 'Client Send')
      MPIUtils.isend(packets, self.server, tag=MPIUtils.core())

class Server:
   def __init__(self, clients):
      self.clients = clients

   def map(self):
      MPIUtils.print(MPIUtils.ALL, 'Server Send')
      for worker in self.clients:
         MPIUtils.print(MPIUtils.ALL, 'Server Send')
         for pack in packets():
            MPIUtils.isend(pack, worker, tag=worker)

   def reduce(self):
      reqs = []
      for client in self.clients:
         MPIUtils.print(MPIUtils.ALL, 'Server Recv')
         reqs.append(MPIUtils.irecv(client, tag=client))
      for req in reqs:
         req.wait()


def test():
   if MPIUtils.core() == MPIUtils.MASTER:
      server = Server([1])
   else:
      client = Client(0)
   while True:
      if MPIUtils.core() == MPIUtils.MASTER:
         server.map()
      elif MPIUtils.core() == 1:
         client.step()
      if MPIUtils.core() == MPIUtils.MASTER:
         server.reduce()

if __name__ == '__main__':
   test()
