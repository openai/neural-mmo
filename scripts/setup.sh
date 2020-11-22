echo "Neural MMO setup; assumes Anaconda Python 3.7 and gcc"
conda install pip

if [[ $1 == "--SERVER_ONLY" ]]; then 
   echo "You have chosen not to install the graphical rendering client"
elif [[ $1 == "" ]]; then
   echo "Installing Neural MMO Client (Unity3D)..."
   git clone --depth=1 https://github.com/jsuarez5341/neural-mmo-client
   mv neural-mmo-client forge/embyr
else
   echo "Specify either --SERVER_ONLY or no argument"
   exit 1
fi

#Install python packages
pip install -r scripts/requirements.txt

python Forge.py generate --sz=1044 --nMaps=256
#python scripts/terrain_api.py generate --sz=1044 --nMaps=256 --invert=True
#python scripts/terrain_api.py generate --sz=84 --nMaps=256 --octaves=1
