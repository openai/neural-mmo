def isInt(x):
   return type(x) in (float, int) and int(x) == x

class CommChannel:
   def __init__(self):
      self.outbox = []
      self.inbox = []

   def put(self, update):
      if len(update) > 0:
         self.outbox.append(update)

   def get(self):
      inbox = self.inbox
      self.inbox = []
      return inbox

   def send(self):
      outbox = self.outbox
      self.outbox = []
      return outbox

   def recv(self, updates):
      self.inbox += updates
