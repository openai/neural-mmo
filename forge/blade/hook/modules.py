from os.path import dirname, basename, isfile
import glob
def modules(f):
   modules = glob.glob(dirname(f)+"/*.py")
   return  [ basename(f)[:-3] for f in modules 
         if isfile(f) and not f.endswith('__init__.py')]


