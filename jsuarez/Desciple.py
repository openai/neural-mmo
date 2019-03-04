class Desciple:
   def __init__(self, entID, config, color=None):
      self.packet = pc.Packet(entID, config, color)
      self.client = pc.Client(self.packet)
      self.server = pc.Server(config)
      self.sync(self.packet, self.server)
      self.sync(self.packet, self.client)

   def sync(self, src, dst):
      for attr in self.packet.paramNames():
         setattr(dst, attr,
            getattr(src, attr))

