from pdb import set_trace as T
import sys, re

assert len(sys.argv) == 2

#A little script for stripping redundant headers
ff = sys.argv[1]
with open(ff, 'r') as f:
   txt = f.read()
   txt = re.sub('Subpackages\n-----------\n', '', txt)
   txt = re.sub('Submodules\n----------\n', '', txt)
with open(ff, 'w') as f:
   f.write(txt)
   
