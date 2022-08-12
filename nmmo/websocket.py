from pdb import set_trace as T
import numpy as np

from signal import signal, SIGINT
import sys, os, json, pickle, time
import threading

from twisted.internet import reactor
from twisted.internet.task import LoopingCall
from twisted.python import log
from twisted.web.server import Site
from twisted.web.static import File

from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol
from autobahn.twisted.resource import WebSocketResource

class GodswordServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super().__init__()
        print("Created a server")
        self.frame = 0

        #"connected" is already used by WSSP
        self.sent_environment = False
        self.isConnected      = False

        self.pos = [0, 0]
        self.cmd = None

    def onOpen(self):
        print("Opened connection to server")

    def onClose(self, wasClean, code=None, reason=None):
        self.isConnected = False
        print('Connection closed')

    def connectionMade(self):
        super().connectionMade()
        self.factory.clientConnectionMade(self)

    def connectionLost(self, reason):
        super().connectionLost(reason)
        self.factory.clientConnectionLost(self)
        self.sent_environment = False

    #Not used without player interaction
    def onMessage(self, packet, isBinary):
        print("Server packet", packet)
        packet    = packet.decode()
        _, packet = packet.split(';') #Strip headeer
        r, c, cmd = packet.split(' ') #Split camera coords
        if len(cmd) == 0 or cmd == '\t':
            cmd = None

        self.pos = [int(r), int(c)]
        self.cmd = cmd

        self.isConnected = True

    def onConnect(self, request):
        print("WebSocket connection request: {}".format(request))
        realm = self.factory.realm
        self.realm = realm
        self.frame += 1

    def serverPacket(self):
        data = self.realm.packet
        return data

    def sendUpdate(self, data):
        packet               = {}
        packet['resource']   = data['resource']
        packet['player']     = data['player']
        packet['npc']        = data['npc']
        packet['pos']        = data['pos']
        packet['wilderness'] = data['wilderness']
        packet['market']     = data['market']
        
        print('Is Connected? : {}'.format(self.isConnected))
        if not self.sent_environment:
            packet['map']    = data['environment']
            packet['border'] = data['border']
            packet['size']   = data['size']

        if 'overlay' in data:
           packet['overlay'] = data['overlay']
           print('SENDING OVERLAY: ', len(packet['overlay']))

        packet = json.dumps(packet).encode('utf8')
        self.sendMessage(packet, False)

class WSServerFactory(WebSocketServerFactory):
    def __init__(self, ip, realm):
        super().__init__(ip)
        self.realm = realm
        self.time = time.time()
        self.clients = []

        self.pos = [0, 0]
        self.cmd = None
        self.tickRate = 0.6
        self.tick = 0

    def update(self, packet):
        self.tick += 1
        uptime = np.round(self.tickRate*self.tick, 1)
        delta = time.time() - self.time
        print('Wall Clock: ', str(delta)[:5], 'Uptime: ', uptime, ', Tick: ', self.tick)
        delta = self.tickRate - delta    
        if delta > 0:
           time.sleep(delta)
        self.time = time.time()

        for client in self.clients:
            client.sendUpdate(packet)
            if client.pos is not None:
                self.pos = client.pos
                self.cmd = client.cmd

        return self.pos, self.cmd

    def clientConnectionMade(self, client):
        self.clients.append(client)

    def clientConnectionLost(self, client):
        self.clients.remove(client)

class Application:
   def __init__(self, realm):
      signal(SIGINT, self.kill)
      log.startLogging(sys.stdout)

      port = 8080
      self.factory          = WSServerFactory(u'ws://localhost:{}'.format(port), realm)
      self.factory.protocol = GodswordServerProtocol 
      resource              = WebSocketResource(self.factory)

      root = File(".")
      root.putChild(b"ws", resource)
      site = Site(root)

      reactor.listenTCP(port, site)

      def run():
          reactor.run(installSignalHandlers=0)

      threading.Thread(target=run).start()

   def update(self, packet):
      return self.factory.update(packet)

   def kill(*args):
      print("Killed by user")
      reactor.stop()
      os._exit(0)
