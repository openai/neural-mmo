
class PhiNet(nn.Module):
   def __init__(self, xdim, h):
      super().__init__()
      self.conv1 = Conv2d(8, int(h/2), 3, stride=2)
      self.conv2 = Conv2d(int(h/2), h, 3, stride=2)
      self.fc1 = torch.nn.Linear(5+4*4*h, h)
      self.fc2 = torch.nn.Linear(h, h)

   def forward(self, conv, flat):
      x = torch.nn.functional.relu(self.conv1(conv))
      x = torch.nn.functional.relu(self.conv2(x))
      x = x.view(-1)

      x = torch.cat((x, flat))
      x = torch.nn.functional.relu(self.fc1(x))
      x = self.fc2(x)

      return x.view(1, -1)

class ForwardsNet(nn.Module):
   def __init__(self, xdim, h, ydim):
      super().__init__()
      self.loss = torch.nn.MSELoss()

      self.fc1 = torch.nn.Linear(NATN+h, h)
      self.fc2 = torch.nn.Linear(h, h)

   def forward(self, atn, phiPrev, phi):
      atnHot = torch.zeros(NATN)
      atnHot.scatter_(0, atn, 1)
      atnHot = atnHot.view(1, -1)

      x = torch.cat((atnHot, phiPrev), 1)
      x = torch.nn.functional.relu(self.fc1(x))
      x = self.fc2(x)

      #Stop grads on phi
      loss = self.loss(x, phi)
      return loss

class BackwardsNet(nn.Module):
   def __init__(self, h, ydim):
      super().__init__()
      self.loss = torch.nn.CrossEntropyLoss()

      self.fc1 = torch.nn.Linear(2*h, h)
      self.fc2 = torch.nn.Linear(h, ydim)

   def forward(self, phiPrev, phi, atn):
      x = torch.cat((phiPrev, phi), 1)
      x = torch.nn.functional.relu(self.fc1(x))
      x = self.fc2(x)

      loss = self.loss(x, atn)
      return loss

class CurNet(nn.Module):
   def __init__(self, xdim, h, ydim):
      super().__init__()
      self.phi = PhiNet(xdim, h)
      self.forwardsDynamics  = ForwardsNet(xdim, h, ydim)
      self.backwardsDynamics = BackwardsNet(h, ydim)

   def forward(self, ents, entID, atn, conv, flat):

      conv = conv.view(1, *conv.size())
      conv = conv.permute(0, 3, 1, 2)

      if entID in ents:
         atn, convPrev, flatPrev = ents[entID]
         phiPrev = self.phi(convPrev, flatPrev)
         phi     = self.phi(conv, flat)

         #Stop both phi's on forward, train both on backward. Confirmed by Harri
         fLoss = self.forwardsDynamics(atn, phiPrev.detach(), phi.detach())
         bLoss = self.backwardsDynamics(phiPrev, phi, atn)

         ri = fLoss.data[0]

         li = 0.20*fLoss + 0.80*bLoss
      else:
         ri, li = 0, torch.tensor(0.0)
         #ri, li = 0, tu.var(np.zeros(1))

      ents[entID] = (atn, conv, flat)
      return ri, li
