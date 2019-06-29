from pdb import set_trace as T
import os

#Install python requirements
print('Installing requirements')
os.system('pip3 install -r scripts/requirements.txt')

#I tried setting this up as a github submodule.
#Never do this ever. They are terrible.
print('Downloading Embyr client')
os.chdir('forge')
os.system('git clone https://github.com/jsuarez5341/neural-mmo-client --recurse-submodules')
os.system('mv neural-mmo-client embyr')
os.chdir('..')

#Build game maps
print('Building game maps')
import terrain
