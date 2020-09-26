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

#Install macOS or Linux specific wheel
if [[ "$OSTYPE" == "darwin"* ]]; then
   pip install -U scripts/ray-0.9.0.dev0-cp37-cp37m-macosx_10_13_intel.whl
else
   pip install -U scripts/ray-0.9.0.dev0-cp37-cp37m-manylinux1_x86_64.whl
fi

python scripts/terrain_api.py generate --sz=1044 --nMaps=256
#python scripts/terrain_api.py generate --sz=1044 --nMaps=256 --invert=True
#python scripts/terrain_api.py generate --sz=84 --nMaps=256 --octaves=1
