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

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)

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
        realm = self.factory.realm
        self.realm = realm
        self.frame += 1

        data = self.serverPacket()
        self.sendUpdate()

    def serverPacket(self):
        data = self.realm.clientData()
        return data

    def sendUpdate(self):
        ent = {}
        data = self.serverPacket()
        entities = data['entities']
        environment = data['environment']
        self.packet['ent'] = entities

        gameMap = environment.np().tolist()
        self.packet['map'] = gameMap

        counts, attention, values = [], [], []
        countColors = []
        for tileList in environment.tiles:
           counts.append([])
           countColors.append([])
           attention.append([])
           values.append([])
           for tile in tileList:
              counts[-1].append(tile.count.value)
              countColors[-1].append(tile.count.color)
              
              attention[-1].append(tile.attention.value)
              values[-1].append(tile.value.value)

        counts    = self.visCounts(counts, countColors)
        attention = self.visVals(attention)
        values    = self.visVals(values)
        globalValues = self.visVals(data['globalValues'])

        self.packet['counts']    = (counts / (np.max(counts))).tolist()
        self.packet['attention'] = attention.tolist()
        self.packet['values']    = values.tolist()
        self.packet['globalValues'] = globalValues.tolist()

        packet = json.dumps(self.packet).encode('utf8')
        self.sendMessage(packet, False)

    #Todo: would be nicer to move this into the javascript,
    #But it would possibly have to go straight into the shader
    def visVals(self, vals, nStd=2):
      vals = np.array(vals)
      R, C = vals.shape
      ary  = np.zeros((R, C, 3))
      vStats = vals[vals != 0]
      vMean = np.mean(vStats)
      vStd  = np.std(vStats)
      for r in range(R):
        for c in range(C):
           val = vals[r, c]
           if val != 0:
              val = (val - vMean) / (nStd * vStd)
              val = np.clip(val+1, 0, 2)/2
              ary[r, c] = [1-val, val, 0]
      return ary

    def visCounts(self, counts, colors, nStd=2):
      counts = np.array(counts)
      colors = np.array(colors)
      R, C = counts.shape
      ary  = np.zeros((R, C, 3))
      vStats = counts[counts!= 0]
      vMean = np.mean(vStats)
      vStd  = np.std(vStats)
      for r in range(R):
        for c in range(C):
           val = counts[r, c]
           color = colors[r, c]
           if val != 0:
              val = (val - vMean) / (nStd * vStd)
              val = np.clip(val+1, 0, 2)/2
              mmax = np.max(color)
              if mmax > 1:
                  color = color / mmax
              ary[r, c] = val * color
      return ary

class WSServerFactory(WebSocketServerFactory):
    def __init__(self, ip, realm, step):
        super().__init__(ip)
        self.realm, self.step = realm, step
        self.clients = []

        self.tickRate = 0.6
        self.tick = 0

        self.step()
        lc = LoopingCall(self.announce)
        lc.start(self.tickRate)

    def announce(self):
        self.tick += 1
        uptime = np.round(self.tickRate*self.tick, 1)
        print('Uptime: ', uptime, ', Tick: ', self.tick)

        for client in self.clients:
            client.sendUpdate()

        self.step()

    def clientConnectionMade(self, client):
        self.clients.append(client)

    def clientConnectionLost(self, client):
        self.clients.remove(client)

class Application:
   def __init__(self, realm, step):
      signal(SIGINT, self.kill)
      self.realm = realm
      log.startLogging(sys.stdout)
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

