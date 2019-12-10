class StimHook:                                                               
   def __init__(self, meta, config):                                          
      self.meta = meta                                                        
      self.config = config                                                    
                                                                              
      self.inputs(meta, config)                                               
                                                                              
   def inputs(self, cls, config):                                             
      for name, c in cls:                                                     
         self.__dict__[c.name] = c(config)                                    
                                                                              
   def outputs(self, config):                                                 
      data = {}                                                               
      for name, cls in self.meta:                                             
         assert type(name) == tuple and len(name) == 1                        
         name       = name[0].lower()                                         
         attr       = self.__dict__[cls.name]                                 
         data[name] = attr.packet()                                           
                                                                              
      return data                                                             
                                                                              
   def packet(self):                                                          
      return self.outputs(self.config)
