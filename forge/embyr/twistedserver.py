from pdb import set_trace as T
import numpy as np

from signal import signal, SIGINT
import sys, os, json, pickle, time
import ray

from twisted.internet import reactor
from twisted.internet.task import LoopingCall
from twisted.python import log
from twisted.web.server import Site
from twisted.web.static import File

from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol
from autobahn.twisted.resource import WebSocketResource

def sign(x):
    return int(np.sign(x))

def move(orig, targ):
    ro, co = orig
    rt, ct = targ
    dr = rt - ro
    dc = ct - co
    if abs(dr) > abs(dc):
        return ro + sign(dr), co
    elif abs(dc) > abs(dr):
        return ro, co + sign(dc)
    else:
        return ro + sign(dr), co + sign(dc)

class GodswordServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super().__init__()
        print("Created a server")
        self.frame = 0
        self.packet = {}

    def onOpen(self):
        print("Opened connection to server")

    def onClose(self, wasClean, code=None, reason=None):
        print('Connection closed')

    def connectionMade(self):
        super().connectionMade()
        self.factory.clientConnectionMade(self)

    def connectionLost(self, reason):
        super().connectionLost(reason)
        self.factory.clientConnectionLost(self)

    #Not used without player interaction
    def onMessage(self, packet, isBinary):
        print("Message", packet)

    def onConnect(self, request):
        print("WebSocket connection request: {}".format(request))
        realm = self.factory.realm.envs[0]
        self.realm = realm
        self.frame += 1

        data = self.serverPacket()
        sz = data['environment'].shape[0]

        self.vals = None
        if data['values'] is not None:
           self.vals = self.visVals(data['values'], sz)

        self.sendUpdate()

    def serverPacket(self):
        data = self.realm.clientData.remote()
        data = ray.get(data)
        data = pickle.loads(data)
        return data

    def sendUpdate(self):
        ent = {}
        data = self.serverPacket()
        entities = data['entities']
        environment = data['environment']
        self.packet['ent'] = entities

        gameMap = environment.np().tolist()
        self.packet['map'] = gameMap

        tiles = []
        for tileList in environment.tiles:
           tl = []
           for tile in tileList:
              tl.append(tile.counts.tolist())
           tiles.append(tl)
        self.packet['counts'] = tiles

        self.packet['values'] = self.vals
        if self.vals is not None:
           self.packet['values'] = self.vals.tolist()
 
        packet = json.dumps(self.packet).encode('utf8')
        self.sendMessage(packet, False)

    #Todo: would be nicer to move this into the javascript,
    #But it would possibly have to go straight into the shader
    def visVals(self, vals, sz):
      ary = np.zeros((sz, sz, 3))
      vMean = np.mean([e[1] for e in vals])
      vStd  = np.std([e[1] for e in vals])
      nStd, nTol = 4.0, 0.5
      grayVal = int(255 / nStd * nTol)
      for v in vals:
         pos, mat = v
         r, c = pos
         mat = (mat - vMean) / vStd
         color = np.clip(mat, -nStd, nStd)
         color = int(color * 255.0 / nStd)
         if color > 0:
             color = (0, color, 128)
         else:
             color = (-color, 0, 128)
         ary[r, c] = color
      return ary.astype(np.uint8)

class WSServerFactory(WebSocketServerFactory):
    def __init__(self, ip, realm, step):
        super().__init__(ip)
        self.realm, self.step = realm, step
        self.clients = []

        self.tickRate = 0.6
        self.tick = 0

        lc = LoopingCall(self.announce)
        lc.start(self.tickRate)

    def announce(self):
        self.tick += 1
        uptime = np.round(self.tickRate*self.tick, 1)
        print('Uptime: ', uptime, ', Tick: ', self.tick)

        if self.tick == 5:
           pass
           #time.sleep(20)
        self.step()
        for client in self.clients:
            client.sendUpdate()

    def clientConnectionMade(self, client):
        self.clients.append(client)

    def clientConnectionLost(self, client):
        self.clients.remove(client)

class Application:
   def __init__(self, realm, step):
      signal(SIGINT, self.kill)
      self.realm = realm
      #log.startLogging(sys.stdout)
      port = 8080

      factory = WSServerFactory(u'ws://localhost:' + str(port), realm, step)
      factory.protocol = GodswordServerProtocol 
      resource         = WebSocketResource(factory)

      # We server static files under "/" and our WebSocket 
      # server under "/ws" (note that Twisted uses bytes 
      # for URIs) under one Twisted Web Site
      root = File(".")
      root.putChild(b"ws", resource)
      site = Site(root)

      reactor.listenTCP(port, site)
      reactor.run()

   def kill(*args):
      print("Killed by user")
      reactor.stop()
      os._exit(0)

