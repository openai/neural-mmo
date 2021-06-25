from pdb import set_trace as T

class StimHook:                                                               
   def __init__(self, meta, config):                                          
      self.meta = meta                                                        
      self.config = config                                                    
                                                                              
      self.inputs(meta, config)                                               
                                                                              
   def inputs(self, cls, config):                                             
      for _, c in cls:                                                     
         self.__dict__[c.name] = c(config)                                    
                                                                              
   def outputs(self, config):                                                 
      data = {}                                                               
      for name, cls in self.meta:                                             
         attr           = self.__dict__[cls.name]
         data[cls.name] = attr.packet()                                           
                                                                              
      return data                                                             
                                                                              
   def packet(self):                                                          
      return self.outputs(self.config)
