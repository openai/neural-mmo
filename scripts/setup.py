from pdb import set_trace as T
import os

#Install python requirements
#print('Installing requirements')
#os.system('pip install -r scripts/requirements.txt')


#curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
#source ~/.poetry/env

#Download the Unity client
print('Downloading Embyr client')
os.chdir('forge')
os.system('git clone https://github.com/jsuarez5341/neural-mmo-client')
os.system('mv -n neural-mmo-client embyr')
os.chdir('..')

#Build game maps
print('Building game maps')
import terrain
